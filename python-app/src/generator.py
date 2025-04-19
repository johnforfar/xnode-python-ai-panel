# ./python-app/src/generator.py
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
from models import Model, ModelArgs, llama3_2_1B
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, ClapModel, MimiModel, AutoFeatureExtractor
import os
import json
from pathlib import Path
import logging
from subprocess import run
from huggingface_hub import hf_hub_download

# Force CPU usage for all operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Use PROJECT_ROOT defined in models.py or redefine if needed
try:
    from models import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
print(f"INFO: [generator.py] PROJECT_ROOT set to: {PROJECT_ROOT}")

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor

# Function to convert WAV to MP3 using ffmpeg
def convert_to_mp3(wav_file, mp3_file):
    try:
        print(f"Converting {wav_file} to {mp3_file}...")
        run(["ffmpeg", "-i", wav_file, "-qscale:a", "2", mp3_file, "-y", "-loglevel", "error"], check=True)
        print(f"Successfully converted to {mp3_file}")
        return True
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return False

class Generator:
    def __init__(self, model: Model, device: str = 'cpu'):
        # --- ADD Logger Initialization ---
        self.logger = logging.getLogger(__name__) # Initialize instance logger
        # --- End Add ---

        self._model = model
        self.device = device

        # Setup CSM model caches (This relies on the passed model instance)
        if hasattr(self._model, 'setup_caches'):
            # The setup_caches call was correctly restored in load_csm_1b,
            # so we don't strictly need to call it *again* here.
            # However, let's keep the check for robustness, but use self.logger
            # self._model.setup_caches(1) # Redundant if called in load_csm_1b
            self.logger.info("Generator received model instance.")
        else:
             self.logger.warning("CSM Model instance passed to Generator does not have setup_caches method.")

        # Load tokenizer
        self.logger.info("Loading text tokenizer (Llama)...")
        self._text_tokenizer = self.load_llama3_tokenizer()
        self.logger.info("Text tokenizer loaded.")

        # Load Mimi using Transformers
        mimi_model_id = "kyutai/mimi"
        self.logger.info(f"Loading Mimi model and feature extractor ({mimi_model_id}) using transformers...")
        try:
            # Load feature extractor (handles resampling, potentially normalization)
            self._mimi_feature_extractor = AutoFeatureExtractor.from_pretrained(mimi_model_id)
            # Load the Mimi model itself
            self._mimi_model = MimiModel.from_pretrained(mimi_model_id).to(self.device)
            self._mimi_model.eval()
            # Get sample rate from feature extractor
            self.sample_rate = self._mimi_feature_extractor.sampling_rate
            self.logger.info(f"Mimi model and feature extractor loaded successfully. Sample rate: {self.sample_rate}")
        except Exception as e:
            self.logger.error(f"Failed to load Mimi model/extractor using transformers: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize Mimi audio tokenizer via transformers.") from e
        # --- End Mimi Loading ---

        # Load Watermarker
        self._watermarker = None; self.watermark_key = None; self._apply_watermark = None
        try:
            from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark as apply_watermark_func
            self._watermarker = load_watermarker(device=device); self.watermark_key = CSM_1B_GH_WATERMARK
            self._apply_watermark = apply_watermark_func; self.logger.info("Watermarker loaded.")
        except ImportError: self.logger.warning("watermarking module not found. Skipping.")
        except Exception as e: self.logger.error(f"Failed to load watermarker: {e}. Skipping.")

        self.logger.info("Generator initialization complete.") # Add final confirmation

    def load_llama3_tokenizer(self):
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        llama_path = os.path.join(PROJECT_ROOT, "models/llama-3.2-1b")
        if not os.path.exists(llama_path):
            raise FileNotFoundError(
                f"Llama tokenizer not found at {llama_path}. "
                "Please download it to './models/llama-3.2-1b/'."
            )
        tokenizer = AutoTokenizer.from_pretrained(llama_path)
        bos = tokenizer.bos_token or "<|begin_of_text|>"
        eos = tokenizer.eos_token or "<|end_of_text|>"
        try:
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0", pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)], )
        except Exception as e: self.logger.warning(f"Could not apply TemplateProcessing: {e}.")
        return tokenizer

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []
        if not text: return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        if not text_tokens: return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)
        text_frame = torch.zeros(len(text_tokens), 33, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(len(text_tokens), 33, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True
        frame_tokens.append(text_frame)
        frame_masks.append(text_frame_mask)
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    @torch.inference_mode()
    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
         if audio.numel() == 0:
              self.logger.warning("Empty audio tensor passed to _tokenize_audio.")
              return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)
         assert audio.ndim == 1, f"Audio must be single channel (1D tensor), got shape {audio.shape}"

         # 1. Preprocess audio using the feature extractor
         # Ensure audio is on CPU for extractor if it doesn't handle device placement
         # The extractor likely handles resampling if needed.
         inputs = self._mimi_feature_extractor(raw_audio=audio.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
         input_values = inputs["input_values"].to(self.device) # Move processed values to target device

         # 2. Encode using the Mimi model
         # Output typically contains audio_codes [B, K, T_frames] and possibly semantic_codes
         encoder_outputs = self._mimi_model.encode(input_values)
         # We likely need the 'audio_codes' attribute
         if not hasattr(encoder_outputs, 'audio_codes'):
              self.logger.error("Mimi encoder output missing 'audio_codes' attribute.")
              raise AttributeError("Mimi encoder output missing 'audio_codes'.")
         # audio_codes shape: [1, num_codebooks, num_frames] (K=num_codebooks=32)
         audio_codes = encoder_outputs.audio_codes.squeeze(0) # Remove batch dim -> [K, T_frames]

         # 3. Format codes into CSM frames (similar to before)
         # Add EOS frame (column of zeros)
         eos_frame = torch.zeros(audio_codes.size(0), 1, device=self.device, dtype=audio_codes.dtype)
         audio_codes_with_eos = torch.cat([audio_codes, eos_frame], dim=1) # Shape becomes [K, T_frames + 1]

         # Transpose to [T_frames + 1, K]
         audio_tokens_t = audio_codes_with_eos.transpose(0, 1)
         num_frames = audio_tokens_t.size(0)
         num_codebooks = audio_tokens_t.size(1)

         # Create frame tensors (size 33: 32 for audio + 1 for text placeholder)
         audio_frame = torch.zeros(num_frames, 33, dtype=torch.long, device=self.device)
         audio_frame_mask = torch.zeros(num_frames, 33, dtype=torch.bool, device=self.device)

         # Place audio tokens into the first slots
         num_codebooks_to_place = min(num_codebooks, 32) # Use actual codebook count from tensor
         audio_frame[:, :num_codebooks_to_place] = audio_tokens_t[:, :num_codebooks_to_place]
         audio_frame_mask[:, :num_codebooks_to_place] = True

         return audio_frame, audio_frame_mask # Return single frame tensor

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        if segment.audio is not None and segment.audio.numel() > 0:
            audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        else:
            audio_tokens = torch.empty((0, 33), dtype=torch.long, device=self.device)
            audio_masks = torch.empty((0, 33), dtype=torch.bool, device=self.device)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self, text: str, speaker: int, context: List[Segment],
        max_audio_length_ms: float = 20_000, temperature: float = 0.9, topk: int = 50,
    ) -> torch.Tensor:
        # --- CSM Frame Generation Loop (Keep as before) ---
        # This part uses self._model (the CSM Model) and its generate_frame method
        if not hasattr(self._model, 'generate_frame'):
             self.logger.error("CSM Model instance does not have the expected 'generate_frame' method.")
             self.logger.warning("Falling back to silent audio generation.")
             num_samples = int(self.sample_rate * (max_audio_length_ms / 1000))
             return torch.zeros(num_samples, device='cpu')

        if hasattr(self._model, 'reset_caches'): self._model.reset_caches()
        max_generation_frames = int(max_audio_length_ms / 80)
        # ... (Tokenization logic using _tokenize_segment remains the same) ...
        tokens, tokens_mask = [], []
        for i, segment in enumerate(context):
             try:
                  segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                  tokens.append(segment_tokens); tokens_mask.append(segment_tokens_mask)
             except Exception as e: self.logger.warning(f"Skipping context segment {i+1}: {e}")
        try:
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
            tokens.append(gen_segment_tokens); tokens_mask.append(gen_segment_tokens_mask)
        except Exception as e: raise ValueError("Failed to tokenize target text.") from e
        if not tokens: raise ValueError("No tokens for generation.")
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = [] # Stores generated frames [1, num_codebooks]
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0).long()

        model_max_seq_len = getattr(getattr(self._model, 'config', None), 'max_seq_len', 2048)
        max_context_len = model_max_seq_len - max_generation_frames
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(f"Input context ({curr_tokens.size(1)}) too long. Max context: {max_context_len}")

        self.logger.info("Starting CSM audio frame generation loop...")
        for i in range(max_generation_frames):
            try:
                # Generate one frame using the CSM model
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0): break
                samples.append(sample)
                # Prepare next input for CSM model
                next_input_frame = torch.zeros(1, 1, 33, dtype=torch.long, device=self.device)
                next_input_mask = torch.zeros(1, 1, 33, dtype=torch.bool, device=self.device)
                num_codebooks = getattr(getattr(self._model, 'config', None), 'audio_num_codebooks', 32)
                next_input_frame[0, 0, :num_codebooks] = sample[0]
                next_input_mask[0, 0, :num_codebooks] = True
                curr_tokens = next_input_frame
                curr_tokens_mask = next_input_mask
                curr_pos = curr_pos[:, -1:] + 1
            except Exception as e:
                 self.logger.error(f"Error during CSM frame generation step {i}: {e}", exc_info=True)
                 break
        # --- End CSM Frame Generation Loop ---

        self.logger.info(f"Generated {len(samples)} audio frames.")
        if not samples: return torch.zeros(0, device='cpu')

        # --- Decode using Transformers Mimi Model ---
        self.logger.info("Decoding audio frames using transformers MimiModel...")
        try:
            # Stack samples: list of [1, num_codebooks] -> tensor [num_frames, 1, num_codebooks]
            # Permute for Mimi decode: [B, K, T_frames] -> [1, num_codebooks, num_frames]
            stacked_samples = torch.stack(samples).permute(1, 2, 0)

            # Decode using the Mimi model's decode method
            # Input should be audio_codes tensor [B, K, T_frames]
            # It might also accept semantic_codes=None if the model uses them
            decoded_output = self._mimi_model.decode(audio_codes=stacked_samples, semantic_codes=None)
            # Output might be a tuple or object, get the audio waveform
            # Assuming the output is directly the waveform [B, T_samples] or has an 'audio_values' attribute
            if isinstance(decoded_output, torch.Tensor):
                 audio = decoded_output.squeeze(0) # Remove batch dim -> [T_samples]
            elif hasattr(decoded_output, 'audio_values'):
                 audio = decoded_output.audio_values.squeeze(0) # Remove batch dim -> [T_samples]
            else:
                  self.logger.error("Unexpected output type from MimiModel.decode.")
                  raise TypeError("Could not extract audio waveform from MimiModel.decode output.")

            self.logger.info(f"Audio decoded successfully via transformers. Shape: {audio.shape}")
        except Exception as e:
             self.logger.error(f"Error decoding audio frames via transformers: {e}", exc_info=True)
             return torch.zeros(0, device='cpu')

        # --- Watermarking (Keep as is) ---
        if self._apply_watermark and self._watermarker and self.watermark_key:
            # ... (watermarking logic) ...
             try:
                 audio_for_wm = audio.to(self._watermarker.device)
                 watermarked_audio, wm_sample_rate = self._apply_watermark(self._watermarker, audio_for_wm, self.sample_rate, self.watermark_key)
                 if wm_sample_rate != self.sample_rate: watermarked_audio = torchaudio.functional.resample(...)
                 audio = watermarked_audio.to(self.device)
             except Exception as e: self.logger.error(f"Error applying watermark: {e}.")


        return audio.cpu() # Return final audio on CPU

def load_csm_1b(device: str = "cpu", model_dir: Path = None) -> Generator:
    """Loads the CSM model locally using Model.from_local_pretrained and returns the Generator."""
    if model_dir is None:
        model_dir = PROJECT_ROOT / "models"
    logger.info(f"Attempting to load CSM-1B model from local directory: {model_dir}")

    csm_model_path = model_dir / "csm-1b" # Path to the directory

    if not csm_model_path.exists():
        logger.error(f"CSM model directory not found: {csm_model_path}")
        raise FileNotFoundError(f"CSM model directory not found at {csm_model_path}")

    # --- Load using Model.from_local_pretrained ---
    try:
        logger.info(f"Calling Model.from_local_pretrained for path: {csm_model_path}")
        # This uses the method defined in models.py which handles safetensors/pytorch loading
        model = Model.from_local_pretrained(str(csm_model_path), device=device)

        # Optional: Convert to bfloat16 if needed
        if device != 'cpu' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
             logger.info("Converting loaded model to bfloat16.")
             model = model.to(torch.bfloat16)

    except Exception as e:
        # Log the specific error from our loading function
        logger.error(f"Failed to load model using Model.from_local_pretrained from {csm_model_path}: {e}", exc_info=True)
        raise # Re-raise the underlying loading error

    # Initialize the Generator class with the loaded model
    try:
        logger.info("Initializing Generator class with loaded model...")
        # Pass the model object (already loaded)
        generator = Generator(model, device=device)
        logger.info("Generator initialized successfully.")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize Generator class after loading model: {e}", exc_info=True)
        raise