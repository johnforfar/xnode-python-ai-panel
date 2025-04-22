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
from collections import OrderedDict
import time # Ensure time is imported for timing
from env import models_dir

# Force CPU usage for all operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)

# --- Import loaders from moshi.models ---
try:
    from moshi.models import loaders # <-- ADD THIS IMPORT
except ImportError:
    logger.error("Failed to import 'loaders' from 'moshi.models'. Please ensure 'moshi-tts' is installed.")
    # Decide how to handle: exit or disable features? Let's allow Generator init to fail later.
    loaders = None # Set to None so checks later fail gracefully if import failed
# --- End loaders import ---

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
            # --- FIX: Determine Cache dtype based on device ---
            if self.device.type == 'cpu':
                cache_dtype = torch.float32 # Use float32 for CPU cache
                logger.info(f"CPU detected. Setting cache dtype to: {cache_dtype}")
            else: # CUDA
                # Use bfloat16 if supported, otherwise float16
                # RuntimeError: Index put requires the source and destination dtypes match
                cache_dtype = torch.float32 # torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                logger.info(f"CUDA detected. Setting cache dtype to: {cache_dtype}")
            # --- End FIX ---

            logger.debug(f"Setting up model caches during Generator init (batch_size=1, device={self.device}, dtype={cache_dtype})...")
            # Ensure model is on the correct device before setting up caches
            self._model.to(self.device)
            # Pass the correctly determined dtype
            self._model.setup_caches(max_batch_size=1, dtype=cache_dtype)
            logger.debug(f"Model caches setup successfully during init.")
        except Exception as e:
             logger.error(f"Error setting up caches during Generator init: {e}", exc_info=True)
             # Depending on severity, you might want to raise this or handle differently

        logger.debug("Loading text tokenizer (Llama)...")
        self._text_tokenizer = self.load_llama3_tokenizer()
        logger.debug("Text tokenizer loaded.")

        # --- Load Mimi Tokenizer using imported loaders ---
        logger.debug(f"Loading Mimi model and feature extractor (kyutai/mimi) to device: {self.device}...")
        try:
            if loaders is None: # Check if import failed
                 raise ImportError("moshi.models.loaders could not be imported.")

            # Pass the device to the loader if possible, or move the model after loading
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME) # Now loaders should be defined
            mimi = loaders.get_mimi(mimi_weight, device=self.device) # Pass device here
            # Ensure mimi components are on the correct device
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
             self.sample_rate = 24000 # Default fallback?
        # --- End Mimi loading ---

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

        # --- Initialize cache using imported OrderedDict ---
        self._stream_buffer_size = 20
        self.max_seq_len = 2048
        self._cache = OrderedDict() # Now OrderedDict should be defined
        self._text_token_cache = {}
        # --- End cache init ---

        logger.debug("Generator initialization complete.")

    def load_llama3_tokenizer(self):
        """
        Loads the Llama-3.2-1B tokenizer, prioritizing a local path.
        """
        # Define the expected local path
        local_tokenizer_path = Path(models_dir()) / "llama-3-2-1b"
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
         if not hasattr(self, '_audio_tokenizer') or self._audio_tokenizer is None:
             logger.error("Audio tokenizer (mimi) not initialized.")
             return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)

         if audio.numel() == 0:
              logger.warning("Empty audio tensor passed to _tokenize_audio.")
              return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)
         assert audio.ndim == 1, f"Audio must be single channel (1D tensor), got shape {audio.shape}"

         try:
            # Use mimi.encode directly
            audio_codes = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0] # Shape [K, T]

            num_codebooks = self._model.config.audio_num_codebooks
            if audio_codes.size(0) != num_codebooks:
                 logger.warning(f"Mimi produced {audio_codes.size(0)} codebooks, but model expects {num_codebooks}. Using expected number.")
                 # Adjust if needed, or error? Let's assume model expects fixed size.
                 # This might need revisiting if Mimi's output varies unexpectedly.
                 if audio_codes.size(0) > num_codebooks:
                     audio_codes = audio_codes[:num_codebooks, :]
                 else: # Pad if fewer? Unlikely with encode.
                     # Handle padding case if necessary, for now assume >= expected
                     pass

            # Add EOS frame (column of zeros)
            eos_frame = torch.zeros(num_codebooks, 1, device=self.device, dtype=audio_codes.dtype)
            audio_codes_with_eos = torch.cat([audio_codes, eos_frame], dim=1) # Shape [K, T_frames + 1]

            # Transpose to [T_frames + 1, K]
            audio_tokens_t = audio_codes_with_eos.transpose(0, 1)
            num_frames = audio_tokens_t.size(0)

            # Create frame tensors (size 33: 32 audio + 1 text)
            frame_size = num_codebooks + 1
            audio_frame = torch.zeros(num_frames, frame_size, dtype=torch.long, device=self.device)
            audio_frame_mask = torch.zeros(num_frames, frame_size, dtype=torch.bool, device=self.device)

            # Place audio tokens - original placed in [:, :-1], let's match that
            audio_frame[:, :-1] = audio_tokens_t # Place K codebooks in first 32 slots
            audio_frame_mask[:, :-1] = True      # Mask those slots

            return audio_frame, audio_frame_mask
         except Exception as e:
             logger.error(f"Error during _tokenize_audio: {e}", exc_info=True)
             return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)

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
        topk: int = 50,
    ):
        start_time_generate = time.perf_counter()
        logger.debug(f"Entering Generator.generate (Original Logic) for speaker {speaker} text: {text[:30]}...")

        # --- Reset KV Cache (as per original) ---
        if hasattr(self._model, 'reset_caches'):
            self._model.reset_caches()
        else:
            logger.warning("Model does not have reset_caches method.")
        # --- End Reset ---

        if torch.cuda.is_available() and self.device.type == 'cuda': # Check if using CUDA
            logger.debug("Clearing CUDA cache before generation.")
            torch.cuda.empty_cache() # Keep cache clearing

        max_generation_len = int(max_audio_length_ms / 80)
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

        # --- Generation Loop (Reverted to Original Sesame Logic) ---
        samples = []
        # Initial inputs for the loop
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        # Global position tracker
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0).long()

        logger.debug(f"Starting frame generation loop (Original Logic, max_len={max_generation_len})...")
        log_interval = 20
        start_time_loop = time.perf_counter()

        for i in range(max_generation_len):
            # loop_iter_start = time.perf_counter() # Optional detailed timing
            try:
                # Generate the next frame using current tokens, masks, and positions
                # generate_frame uses KV cache internally based on input_pos
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                # sample shape: [B, 32] (num_codebooks)

                if torch.all(sample == 0):
                    logger.info(f"Termination condition detected at frame {i+1}.")
                    break
                samples.append(sample) # Append the generated frame [1, 32]

                # Prepare input for the *next* iteration (Original Sesame Logic)
                # Input is the frame just generated, formatted correctly.
                next_token_frame = torch.zeros(1, 1, self._model.config.audio_num_codebooks + 1, dtype=torch.long, device=self.device)
                next_token_frame_mask = torch.zeros(1, 1, self._model.config.audio_num_codebooks + 1, dtype=torch.bool, device=self.device)

                # Place sample into audio slots (first 32)
                next_token_frame[0, 0, :self._model.config.audio_num_codebooks] = sample[0]
                next_token_frame_mask[0, 0, :self._model.config.audio_num_codebooks] = True

                # Update variables for next loop iteration
                curr_tokens = next_token_frame         # Next input token frame
                curr_tokens_mask = next_token_frame_mask # Next input mask frame
                curr_pos = curr_pos[:, -1:] + 1      # Increment global position

                # --- Progress Logging ---
                if (i + 1) % log_interval == 0:
                     # loop_iter_end = time.perf_counter() # Optional
                     elapsed_loop = time.perf_counter() - start_time_loop
                     frames_per_sec = (i + 1) / elapsed_loop if elapsed_loop > 0 else float('inf')
                     logger.info(f"Generated frame {i+1}/{max_generation_len} ({frames_per_sec:.2f} frames/sec so far)")
                # --- End Progress Logging ---

            except Exception as frame_e:
                logger.error(f"Error during frame generation {i+1}: {frame_e}", exc_info=True)
                break
        # --- End Generation Loop (Reverted) ---

        end_time_loop = time.perf_counter()
        logger.debug(f"Frame generation loop finished in {end_time_loop - start_time_loop:.2f} seconds. Generated {len(samples)} frames.")

        if not samples:
            logger.warning("No audio frames were generated.")
            return torch.tensor([]) # Return empty tensor if no samples

        logger.debug("Decoding generated frames...")
        if not hasattr(self, '_audio_tokenizer') or self._audio_tokenizer is None:
             logger.error("Audio tokenizer not found in Generator. Cannot decode frames.")
             return torch.tensor([])

        try:
            # --- Decoding (Keep aligned with Fork/Original) ---
            if not samples:
                 logger.warning("No samples generated for decoding.")
                 return torch.tensor([])

            # Stack samples: List of [1, 32] -> [len(samples), 1, 32]
            stacked_samples = torch.stack(samples)
            # Permute for decoder: [len(samples), 1, 32] -> [1, 32, len(samples)] (B, K, T)
            stacked_samples = stacked_samples.permute(1, 2, 0)

            # Decode with all codebooks (consistent with original non-streaming decode)
            decoded_audio = self._audio_tokenizer.decode(stacked_samples)
            # --- End Decoding ---

            # Squeeze unnecessary dimensions (batch and channel if mono)
            decoded_audio = decoded_audio.squeeze(0).squeeze(0)
            logger.debug("Frames decoded successfully.")

            end_time_generate = time.perf_counter()
            logger.info(f"Generator.generate completed in {end_time_generate - start_time_generate:.2f} seconds total.")
            return decoded_audio.cpu()
        except Exception as decode_e:
            logger.error(f"Error decoding audio frames: {decode_e}", exc_info=True)
            return torch.tensor([]) # Return empty on decoding error

def load_csm_1b_local(model_path: str, device: str = "cuda", audio_num_codebooks: int = 32):
    """
    Load the CSM-1B model from a local checkpoint path.
    model_path should be the BASE models directory (e.g., /path/to/models)
    """
    target_device = torch.device(device)
    csm_specific_path = Path(model_path) / "csm-1b"
    logger.info(f"Attempting to load CSM-1B model from local directory: {csm_specific_path} (will move to {target_device} after loading)")

    if not csm_specific_path.is_dir():
         logger.error(f"CSM local model directory not found: {csm_specific_path}")
         raise FileNotFoundError(f"CSM local model directory not found: {csm_specific_path}")

    try:
        model = Model.from_local_pretrained(str(csm_specific_path))
        logger.info(f"Moving loaded model to target device: {target_device}")
        model = model.to(target_device)
        model.eval()
        logger.info(f"Model moved to {target_device} and set to eval mode.")
    except Exception as e:
        logger.error(f"Failed during Model.from_local_pretrained or device transfer for CSM: {e}", exc_info=True)
        raise

    logger.info("CSM Model loaded and moved. Creating generator...")
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