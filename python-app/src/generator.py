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
            # --- FORCE FLOAT32 for cache ---
            cache_dtype = torch.float32
            logger.warning(f"FORCING FLOAT32 for cache dtype (Device: {self.device.type}).")
            # --- END FORCE FLOAT32 ---
            # if self.device.type == 'cpu':
            #     cache_dtype = torch.float32 # Use float32 for CPU cache
            #     logger.info(f"CPU detected. Setting cache dtype to: {cache_dtype}")
            # else: # CUDA
            #     # Use bfloat16 if supported, otherwise float16 (more standard for GPU)
            #     if torch.cuda.is_bf16_supported():
            #         cache_dtype = torch.bfloat16
            #         logger.info(f"CUDA detected. Setting cache dtype to: {cache_dtype} (bf16 supported)")
            #     else:
            #         cache_dtype = torch.float16
            #         logger.info(f"CUDA detected. Setting cache dtype to: {cache_dtype} (bf16 NOT supported)")

            logger.debug(f"Setting up model caches during Generator init (batch_size=1, device={self.device}, dtype={cache_dtype})...")
            self._model.to(self.device) # Ensure model is on device before setup_caches
            # Pass the potentially changed dtype
            self._model.setup_caches(max_batch_size=1, dtype=cache_dtype)
            logger.debug(f"Model caches setup successfully during init.")
        except Exception as e:
             logger.error(f"Error setting up caches during Generator init: {e}", exc_info=True)
             # Consider re-raising or handling more gracefully if setup fails

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
                 if audio_codes.size(0) > num_codebooks:
                     audio_codes = audio_codes[:num_codebooks, :]
                 # else: Handle padding if necessary

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

            # Place audio tokens
            audio_frame[:, :-1] = audio_tokens_t
            audio_frame_mask[:, :-1] = True

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
        target_device = self.device
        logger.debug(f"Entering Generator.generate for speaker {speaker} on device {target_device} text: {text[:30]}...")

        if target_device == "cuda":
            torch.cuda.empty_cache()

        max_generation_len = int(max_audio_length_ms / 80)
        max_seq_len = 2048 # Set based on Llama-3.2-1B backbone config

        # --- Tokenization logic ---
        logger.debug("Tokenizing context and text...")
        tokens_segments: List[Tuple[torch.Tensor, torch.Tensor]] = []
        try:
             # Tokenize context segments
             tokens_segments = [self._tokenize_segment(s) for s in context]
             # Tokenize the current text segment
             tokens_segments.append(self._tokenize_text_segment(text, speaker))
             logger.debug(f"Tokenized {len(tokens_segments)} segments.")
        except Exception as token_e:
             logger.error(f"Error during tokenization: {token_e}", exc_info=True)
             return torch.tensor([]) # Return empty tensor on tokenization error

        # --- FIX: Replace _stack_tokens with direct concatenation ---
        # Initialize empty tensors on the target device
        prompt_tokens = torch.empty((0, self._model.config.audio_num_codebooks + 1), dtype=torch.long, device=target_device)
        prompt_tokens_mask = torch.empty((0, self._model.config.audio_num_codebooks + 1), dtype=torch.bool, device=target_device)

        try:
            if tokens_segments: # Check if there are any segments to concatenate
                 logger.debug("Concatenating token segments...")
                 # Concatenate tensors from the list of tuples
                 all_segment_tokens = [seg[0] for seg in tokens_segments]
                 all_segment_masks = [seg[1] for seg in tokens_segments]
                 prompt_tokens = torch.cat(all_segment_tokens, dim=0)
                 prompt_tokens_mask = torch.cat(all_segment_masks, dim=0)
                 logger.debug(f"Concatenated tokens shape: {prompt_tokens.shape}")
            else:
                logger.warning("No token segments found after tokenization.")

            # Ensure tensors are on the correct device (might be redundant but safe)
            prompt_tokens = prompt_tokens.to(target_device)
            prompt_tokens_mask = prompt_tokens_mask.to(target_device)

        except Exception as concat_e:
             logger.error(f"Error concatenating token segments: {concat_e}", exc_info=True)
             return torch.tensor([])
        # --- End FIX ---

        # --- FIX: Correct BOS token logic ---
        bos_frame = torch.zeros(1, self._model.config.audio_num_codebooks + 1, dtype=torch.long, device=target_device)
        bos_frame_mask = torch.zeros(1, self._model.config.audio_num_codebooks + 1, dtype=torch.bool, device=target_device)
        # Check tokenizer for BOS ID
        if self._text_tokenizer and hasattr(self._text_tokenizer, 'bos_token_id') and self._text_tokenizer.bos_token_id is not None:
             bos_id = self._text_tokenizer.bos_token_id
             # Assuming bos_id applies to the last dimension (text token)
             bos_frame[0, -1] = bos_id
             bos_frame_mask[0, -1] = True
             # Prepend the created BOS frame
             prompt_tokens = torch.cat((bos_frame, prompt_tokens), dim=0)
             prompt_tokens_mask = torch.cat((bos_frame_mask, prompt_tokens_mask), dim=0)
             logger.debug(f"Prepended BOS token (ID: {bos_id}). New prompt shape: {prompt_tokens.shape}")
        else:
             logger.warning("Text tokenizer does not have bos_token_id, cannot prepend BOS frame.")
        # --- End FIX ---

        # Check max_seq_len again AFTER concatenation and BOS and handle truncation if needed
        if prompt_tokens.size(0) > max_seq_len:
             logger.warning(f"Prompt length ({prompt_tokens.size(0)}) exceeds max_seq_len ({max_seq_len}). Truncating.")
             prompt_tokens = prompt_tokens[-max_seq_len:]
             prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]

        # --- Generation Loop ---
        samples = []
        # Ensure curr_tokens and mask start on the correct device
        curr_tokens = prompt_tokens.unsqueeze(0).to(target_device) # Redundant .to() is safe
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0).to(target_device) # Redundant .to() is safe
        # Ensure curr_pos is created on the correct device
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=target_device).unsqueeze(0).long()

        MIN_FRAMES_BEFORE_TERMINATION = 25
        logger.debug(f"Starting frame generation loop (max_len={max_generation_len}, min_frames={MIN_FRAMES_BEFORE_TERMINATION})...")
        log_interval = 20
        start_time_loop = time.perf_counter()

        for i in range(max_generation_len):
            try:
                # --- Ensure model and inputs are on the same device before call ---
                # (Model should already be on target_device from loading)
                # (curr_tokens, mask, pos are confirmed/moved above or created on target_device)
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)

                # sample will be on the same device as the model output (target_device)

                if i >= MIN_FRAMES_BEFORE_TERMINATION and torch.all(sample == 0):
                    logger.info(f"Termination condition detected at frame {i+1} (after min threshold).")
                    break

                samples.append(sample) # Appending tensors on target_device

                # Create next inputs directly on the target device
                next_token_frame = torch.zeros(1, 1, self._model.config.audio_num_codebooks + 1, dtype=torch.long, device=target_device)
                next_token_frame_mask = torch.zeros(1, 1, self._model.config.audio_num_codebooks + 1, dtype=torch.bool, device=target_device)
                next_token_frame[0, 0, :self._model.config.audio_num_codebooks] = sample[0] # sample is on target_device
                next_token_frame_mask[0, 0, :self._model.config.audio_num_codebooks] = True
                curr_tokens = next_token_frame
                curr_tokens_mask = next_token_frame_mask
                # curr_pos update should also be on target_device implicitly
                curr_pos = curr_pos[:, -1:] + 1

                if (i + 1) % log_interval == 0:
                     elapsed = time.perf_counter() - start_time_loop
                     logger.debug(f"Generated frame {i+1}/{max_generation_len} ({elapsed:.2f}s elapsed)")

            except Exception as frame_e:
                logger.error(f"Error during frame generation {i+1}: {frame_e}", exc_info=True)
                # If error is device mismatch, log devices
                if "Expected all tensors to be on the same device" in str(frame_e):
                     logger.error(f"Device state: Model on {next(self._model.parameters()).device}, curr_tokens on {curr_tokens.device}, curr_tokens_mask on {curr_tokens_mask.device}, curr_pos on {curr_pos.device}")
                break # Stop generation on error
        # --- End Generation Loop ---

        # ... (Rest of the function: timing, decoding, return - Ensure decode happens on CPU) ...
        end_time_loop = time.perf_counter()
        logger.debug(f"Frame generation loop finished in {end_time_loop - start_time_loop:.2f} seconds. Generated {len(samples)} frames.")

        if not samples:
            logger.warning("No audio frames were generated.")
            return torch.tensor([]) # Return empty CPU tensor

        logger.debug("Decoding generated frames...")
        if not hasattr(self, '_audio_tokenizer') or self._audio_tokenizer is None:
             logger.error("Audio tokenizer not found in Generator. Cannot decode frames.")
             return torch.tensor([])

        try:
            # Stack tensors (will be on target_device)
            stacked_samples = torch.stack(samples)
            stacked_samples = stacked_samples.permute(1, 2, 0)

            # --- Decoding often happens on CPU, ensure input is moved ---
            # (Assuming _audio_tokenizer.decode expects CPU tensors)
            decoded_audio = self._audio_tokenizer.decode(stacked_samples.cpu())
            # --- End CPU move ---
            decoded_audio = decoded_audio.squeeze(0).squeeze(0)
            logger.debug("Frames decoded successfully.")

            end_time_generate = time.perf_counter()
            logger.info(f"Generator.generate completed in {end_time_generate - start_time_generate:.2f} seconds total.")
            # Return final audio on CPU
            return decoded_audio.cpu()
        except Exception as decode_e:
            logger.error(f"Error decoding audio frames: {decode_e}", exc_info=True)
            return torch.tensor([]) # Return empty CPU tensor

def load_csm_1b_local(model_path: str, device: str = "cuda", audio_num_codebooks: int = 32):
    """
    Load the CSM-1B model from a local checkpoint path, aligning with original repo dtype handling.
    model_path should be the BASE models directory (e.g., /path/to/models)
    """
    target_device = torch.device(device)
    csm_specific_path = Path(model_path) / "csm-1b"
    logger.info(f"Attempting to load CSM-1B model from local directory: {csm_specific_path} (will move to {target_device} after loading)")

    if not csm_specific_path.is_dir():
         logger.error(f"CSM local model directory not found: {csm_specific_path}")
         raise FileNotFoundError(f"CSM local model directory not found: {csm_specific_path}")

    try:
        # Load to CPU first
        model = Model.from_local_pretrained(str(csm_specific_path))

        # --- ALIGNMENT: Determine target dtype and cast model globally during device move ---
        logger.info(f"Moving loaded model to target device: {target_device} and casting global dtype...")
        # Determine target dtype based on device and support
        # --- FORCE FLOAT32 ---
        target_dtype = torch.float32
        logger.warning(f"FORCING FLOAT32 dtype ({target_device.type}).")
        # --- END FORCE FLOAT32 ---
        # if target_device.type == 'cuda':
        #     if torch.cuda.is_bf16_supported():
        #         target_dtype = torch.bfloat16
        #         logger.info("Targeting bfloat16 dtype (CUDA).")
        #     else:
        #         target_dtype = torch.float16 # Fallback to float16 if bf16 not supported
        #         logger.info("Targeting float16 dtype (CUDA, bf16 not supported).")
        # else: # CPU
        #     target_dtype = torch.float32 # Use float32 for CPU
        #     logger.info("Targeting float32 dtype (CPU).")

        # Move model to target device AND set its global default dtype
        model = model.to(device=target_device, dtype=target_dtype)
        # --- End ALIGNMENT ---

        model.eval()
        # --- FIX: Check parameter dtype for logging ---
        log_dtype = next(model.parameters()).dtype if list(model.parameters()) else 'N/A' # Handle potential case of no parameters
        logger.info(f"Model moved to {target_device} (first param dtype: {log_dtype}) and set to eval mode.")
        # --- End FIX ---

    except Exception as e:
        logger.error(f"Failed during Model.from_local_pretrained or device/dtype transfer for CSM: {e}", exc_info=True)
        raise

    logger.info("CSM Model loaded and moved. Creating generator...")
    # Pass the actual device object to the Generator
    generator = Generator(model, device=target_device) # Generator init determines cache_dtype based on support
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