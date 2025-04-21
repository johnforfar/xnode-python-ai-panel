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

# --- Define project root path within generator.py as well ---
# Needed to construct the local model path
try:
    GEN_APP_DIR = Path(__file__).resolve().parent
    GEN_PROJECT_ROOT = GEN_APP_DIR.parent.parent
except NameError:
    # Fallback if __file__ isn't defined (e.g., interactive session)
    GEN_PROJECT_ROOT = Path('.').resolve().parent.parent
    print("Warning: __file__ not defined, assuming standard project structure.")
# --- End project root definition ---

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
    def __init__(self, model: Model, device: torch.device):
        logger.debug(f"Initializing Generator class on device: {device}...")
        self._model = model
        self.device = device
        logger.debug(f"Generator device set to: {self.device}")

        # Setup caches ONCE during initialization
        try:
            logger.debug(f"Setting up model caches during Generator init (batch_size=1, device={self.device})...")
            dtype = torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
            self._model.to(self.device)
            self._model.setup_caches(max_batch_size=1, dtype=dtype)
            logger.debug(f"Model caches setup successfully during init.")
        except Exception as e:
             logger.error(f"Error setting up caches during Generator init: {e}", exc_info=True)

        logger.debug("Loading text tokenizer (Llama)...")
        self._text_tokenizer = self.load_llama3_tokenizer()
        logger.debug("Text tokenizer loaded.")

        logger.debug(f"Loading Mimi model and feature extractor (kyutai/mimi) to device: {self.device}...")
        try:
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            mimi = loaders.get_mimi(mimi_weight, device=self.device)
            mimi = mimi.to(self.device)

            num_codebooks = model.config.audio_num_codebooks
            mimi.set_num_codebooks(num_codebooks)
            self._num_codebooks = num_codebooks
            self._audio_tokenizer = mimi
            self.sample_rate = mimi.sample_rate
            logger.debug(f"Mimi model and feature extractor loaded successfully. Sample rate: {self.sample_rate}")
        except Exception as e:
             logger.error(f"Failed to load or setup Mimi audio tokenizer: {e}", exc_info=True)
             self._audio_tokenizer = None
             self.sample_rate = 24000

        # Optional: Load watermarker if used (ensure device placement)
        try:
            logger.debug(f"Loading Watermarker to device: {self.device}...")
            from silentcipher import Watermarker
            self.watermarker = Watermarker(device=self.device)
            logger.debug("Watermarker loaded.")
        except ImportError:
            logger.warning("silentcipher not found, watermarking disabled.")
            self.watermarker = None
        except Exception as e:
            logger.error(f"Failed to load watermarker: {e}", exc_info=True)
            self.watermarker = None

        self._stream_buffer_size = 20
        self.max_seq_len = 2048
        self._cache = OrderedDict()
        self._text_token_cache = {}

        logger.debug("Generator initialization complete.")

    def load_llama3_tokenizer(self):
        """
        Loads the Llama-3.2-1B tokenizer, prioritizing a local path.
        """
        # Define the expected local path
        local_tokenizer_path = GEN_PROJECT_ROOT / "models" / "llama-3-2-1b"
        hub_identifier = "unsloth/Llama-3.2-1B" # Or meta-llama/Meta-Llama-3.1-8B-Instruct

        tokenizer = None
        try:
            if local_tokenizer_path.is_dir():
                logger.info(f"Attempting to load Llama tokenizer from local path: {local_tokenizer_path}")
                tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer_path))
                logger.info("Successfully loaded Llama tokenizer from local path.")
            else:
                 logger.info(f"Local Llama tokenizer path not found: {local_tokenizer_path}. Attempting Hub download.")
        except Exception as e:
            logger.warning(f"Failed to load Llama tokenizer from local path {local_tokenizer_path}: {e}. Attempting Hub download.")
            tokenizer = None # Ensure tokenizer is None if local loading failed

        if tokenizer is None:
            try:
                logger.info(f"Attempting to load Llama tokenizer from Hub: {hub_identifier}")
                # Environment variables should guide this to check cache first
                tokenizer = AutoTokenizer.from_pretrained(hub_identifier)
                logger.info(f"Successfully loaded Llama tokenizer from Hub/cache ({hub_identifier}).")
            except Exception as e:
                logger.error(f"CRITICAL: Failed to load Llama tokenizer from both local path and Hub: {e}", exc_info=True)
                raise # Re-raise the error if loading fails completely

        # Apply post-processor (remains the same)
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        if bos and eos: # Ensure tokens exist before setting processor
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
            )
        else:
            logger.warning("BOS or EOS token not found in tokenizer, cannot set TemplateProcessing.")

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
         inputs = self._audio_tokenizer.feature_extractor(raw_audio=audio.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
         input_values = inputs["input_values"].to(self.device) # Move processed values to target device

         # 2. Encode using the Mimi model
         # Output typically contains audio_codes [B, K, T_frames] and possibly semantic_codes
         encoder_outputs = self._audio_tokenizer.model.encode(input_values)
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
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.8,
        topk: int = 40,
    ):
        logger.debug(f"Entering Generator.generate for speaker {speaker} text: {text[:30]}...")

        if torch.cuda.is_available() and self.device.type == 'cuda': # Check if using CUDA
            logger.debug("Clearing CUDA cache before generation.")
            torch.cuda.empty_cache() # Keep cache clearing

        max_generation_len = int(max_audio_length_ms / 80) # Assuming 1 frame = 80ms
        tokens, tokens_mask = [], []

        # ... (Tokenization logic - unchanged) ...
        logger.debug("Tokenizing context and input text...")
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        max_seq_len = 2048 # Or self.max_seq_len if defined
        if prompt_tokens.size(0) > max_seq_len:
            logger.warning(f"Prompt length ({prompt_tokens.size(0)}) exceeds max_seq_len ({max_seq_len}). Truncating.")
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0).long() # Correct device

        samples = []
        logger.debug(f"Starting frame generation loop (max_len={max_generation_len})...")
        # Ensure audio tokenizer is set up for streaming if needed by decode
        # Although decode is called after the loop, setup might be needed earlier
        # Let's assume the original code handled this correctly or decode doesn't require it.

        # Determine appropriate dtype for autocast based on the Generator's device
        autocast_enabled = self.device.type == 'cuda' # Check if CUDA is the device type
        dtype = torch.bfloat16 if autocast_enabled and torch.cuda.is_bf16_supported() else torch.float16
        logger.debug(f"Autocast enabled: {autocast_enabled}, dtype: {dtype}")

        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=autocast_enabled):
             for i in range(max_generation_len):
                 try:
                    # Log current position being processed
                    # logger.debug(f"Generating frame {i+1}/{max_generation_len} for position {curr_pos.item() if curr_pos.numel() == 1 else curr_pos.shape}")

                    # --- Ensure generate_frame uses the correct autocast context ---
                    # The outer `with torch.autocast` should cover this call
                    sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                    # --- End generate_frame call ---

                    if torch.all(sample == 0):
                        logger.info(f"Detected termination condition (all zeros sample) at frame {i+1}.")
                        break
                    samples.append(sample)

                    # Update tokens and position for the next iteration
                    # Ensure updates happen on the correct device
                    next_sample_token = sample.to(self.device)
                    next_mask = torch.ones_like(next_sample_token).bool().to(self.device)
                    eos_token = torch.zeros(1, 1, device=self.device).long()
                    eos_mask = torch.zeros(1, 1, device=self.device).bool()

                    # Combine the generated sample with EOS for next input step
                    curr_tokens = torch.cat([next_sample_token, eos_token], dim=1).unsqueeze(1)
                    curr_tokens_mask = torch.cat([next_mask, eos_mask], dim=1).unsqueeze(1)
                    curr_pos = curr_pos[:, -1:] + 1 # Increment position

                 except Exception as frame_e:
                      logger.error(f"Error during frame generation {i+1}: {frame_e}", exc_info=True)
                      break # Stop generation on error

        logger.debug(f"Frame generation loop finished. Generated {len(samples)} frames.")

        if not samples:
            logger.warning("No audio frames were generated.")
            return torch.tensor([]) # Return empty tensor if no samples

        logger.debug("Decoding generated frames...")
        # Decode collected frames
        # Need to ensure _audio_tokenizer is available and correctly configured
        if not hasattr(self, '_audio_tokenizer') or self._audio_tokenizer is None:
             logger.error("Audio tokenizer not found in Generator. Cannot decode frames.")
             return torch.tensor([])

        try:
            # Permute dimensions for decoder: (batch, codebooks, sequence_length) -> (1, num_codebooks, len(samples))
            stacked_samples = torch.stack(samples).permute(1, 2, 0)
            # Ensure it's on the correct device for the tokenizer
            stacked_samples = stacked_samples.to(self._audio_tokenizer.device if hasattr(self._audio_tokenizer, 'device') else self.device)
            decoded_audio = self._audio_tokenizer.decode(stacked_samples)
            # Squeeze unnecessary dimensions (batch and channel if mono)
            decoded_audio = decoded_audio.squeeze(0).squeeze(0)
            logger.debug("Frames decoded successfully.")
            return decoded_audio.cpu() # Ensure output is on CPU
        except Exception as decode_e:
            logger.error(f"Error decoding audio frames: {decode_e}", exc_info=True)
            return torch.tensor([]) # Return empty on decoding error

def load_csm_1b_local(model_path: str, device: str = "cuda", audio_num_codebooks: int = 32):
    """
    Load the CSM-1B model from a local checkpoint path.
    model_path should be the BASE models directory (e.g., /path/to/models)
    """
    target_device = torch.device(device)
    # --- Construct specific path INSIDE this function ---
    csm_specific_path = Path(model_path) / "csm-1b"
    # --- End construction ---
    logger.info(f"Attempting to load CSM-1B model from local directory: {csm_specific_path} onto device: {target_device}")

    if not csm_specific_path.is_dir():
         logger.error(f"CSM local model directory not found: {csm_specific_path}")
         raise FileNotFoundError(f"CSM local model directory not found: {csm_specific_path}")

    try:
        # Pass the SPECIFIC path to the loader
        model = Model.from_local_pretrained(str(csm_specific_path), target_device=target_device)
        model.eval()
        model = model.to(target_device)
    except Exception as e:
        logger.error(f"Failed during Model.from_local_pretrained for CSM: {e}", exc_info=True)
        raise

    logger.info("CSM Model loaded. Creating generator...")
    generator = Generator(model, device=target_device)
    return generator

def load_csm_1b(device: str = "cuda", model_dir: Path | str | None = None) -> Generator:
    """
    Load the CSM-1B model from Hub or local cache.
    """
    target_device = torch.device(device)
    logger.info(f"Attempting to load CSM-1B model from Hub/cache onto device: {target_device}")

    try:
        model = Model.from_pretrained("sesame/csm-1b", cache_dir=str(model_dir) if model_dir else None)
        model.eval()
        model = model.to(target_device)

        if target_device.type == 'cuda':
            logger.info("Applying torch.compile optimizations (this might take a moment)...")
            logger.info("Torch compile skipped for now (enable if needed).")
        else:
            logger.info("Torch compile skipped (not on CUDA device).")

    except Exception as e:
        logger.error(f"Failed during Model.from_pretrained('sesame/csm-1b'): {e}", exc_info=True)
        raise

    logger.info("Model loaded successfully. Creating generator...")
    generator = Generator(model, device=target_device)
    return generator