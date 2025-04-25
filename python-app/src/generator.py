# generator.py
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from pathlib import Path
import logging
import time

# --- Local Imports ---
# Uses the models.py aligned with the original CSM repo
from models import Model, ModelArgs
# Import watermarking functions/constants from the original CSM repo's watermarking.py
# Make sure watermarking.py exists in the same directory or adjust import path
try:
    from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
    watermarking_available = True
except ImportError:
    logging.warning("watermarking.py not found or failed to import. Watermarking will be disabled.")
    watermarking_available = False
    # Define dummy values if needed, or handle absence later
    CSM_1B_GH_WATERMARK = None
    load_watermarker = None
    watermark = None

# Import environment functions for paths
from env import models_dir as models_dir_env, data_dir
# --- End Local Imports ---

# --- Transformer/Tokenizer Imports ---
try:
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: 'tokenizers' or 'transformers' library not found. Please install them.")
    raise
# --- End Transformer/Tokenizer Imports ---

# --- Moshi Import for Audio Tokenizer ---
try:
    # Adjust if your moshi installation uses a different structure
    from moshi.models import loaders
except ImportError:
    print("ERROR: Failed to import 'loaders' from 'moshi.models'. Please ensure 'moshi-tts' is installed correctly.")
    loaders = None # Allow Generator init to fail if loaders are needed and missing
# --- End Moshi Import ---

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    """Represents a segment of speech with speaker, text, and audio.
       Matches the structure used in the original CSM repo's run_csm.py and generator.py.
    """
    speaker: int
    text: str
    # Expects audio as a 1D tensor at the target sample rate (e.g., 24_000) on the correct device
    audio: torch.Tensor | None # Allow None for text-only segments (e.g., the segment being generated)

# --- Llama Tokenizer Loading (adapted for local preference, matches original template) ---
def load_llama3_tokenizer(model_base_dir: Path | str | None = None):
    """
    Loads the Llama-3.2-1B tokenizer, prioritizing a local path.
    Uses 'meta-llama/Llama-3.2-1B' as the Hub identifier.
    Applies the specific template processing from the original CSM repo.
    """
    hub_identifier = "meta-llama/Llama-3.2-1B" # Primary target
    local_subdir_name = "Llama-3.2-1B" # Expected local subdirectory

    tokenizer = None
    if model_base_dir:
        local_tokenizer_path = Path(model_base_dir) / local_subdir_name
        if local_tokenizer_path.is_dir():
            logger.info(f"Attempting to load Llama tokenizer from local path: {local_tokenizer_path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer_path))
                logger.info("Successfully loaded Llama tokenizer from local path.")
            except Exception as e:
                logger.warning(f"Failed to load Llama tokenizer from local path {local_tokenizer_path}: {e}. Attempting Hub download.")
                tokenizer = None
        else:
            # Only log if model_base_dir was actually provided
            logger.info(f"Local Llama tokenizer path not found: {local_tokenizer_path}. Attempting Hub download.")

    if tokenizer is None:
        logger.info(f"Attempting to load Llama tokenizer from Hub: {hub_identifier}")
        try:
            # Use model_base_dir as cache_dir for downloads if specified
            cache_dir = str(model_base_dir) if model_base_dir else None
            tokenizer = AutoTokenizer.from_pretrained(hub_identifier, cache_dir=cache_dir)
            logger.info(f"Successfully loaded Llama tokenizer from Hub/cache ('{hub_identifier}').")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load Llama tokenizer from both local path and Hub: {e}", exc_info=True)
            raise # Loading tokenizer is critical

    # Apply Template Processing (Exact match to original CSM repo)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    if bos and eos and hasattr(tokenizer, 'bos_token_id') and hasattr(tokenizer, 'eos_token_id'):
        try:
            # This specific post-processor setup is from the original CSM repo
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
            )
            logger.debug("Applied original CSM TemplateProcessing to Llama tokenizer.")
        except Exception as e:
            logger.error(f"Failed to apply TemplateProcessing to tokenizer: {e}", exc_info=True)
            # This might be critical depending on how the model was trained
            logger.warning("Continuing without TemplateProcessing. Model behavior might be affected.")
    else:
        logger.warning("BOS or EOS token/ID not found in tokenizer, cannot set TemplateProcessing.")

    return tokenizer
# --- End Llama Tokenizer Loading ---

class Generator:
    """Handles audio generation using a CSM Model, closely following original CSM repo logic."""
    def __init__(self, model: Model, device: str | torch.device = "cuda", model_base_dir: Path | str | None = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing Generator class (CSM Aligned)...")

        # Ensure model is on the target device before proceeding
        self.device = torch.device(device if isinstance(device, str) else str(device))
        self._model = model.to(self.device)
        self.logger.info(f"Generator using device: {self.device}. Model moved to device.")

        # Determine appropriate dtype for caches based on device and support
        # This matches the logic from the original CSM loading which used bfloat16 if available
        if self.device.type == 'cuda':
            # Use bfloat16 if supported (common on modern GPUs), otherwise float16
            cache_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.logger.info(f"Using {cache_dtype} for KV cache on CUDA.")
        else: # CPU or other devices
            cache_dtype = torch.float32
            self.logger.info(f"Using float32 for KV cache on {self.device.type}.")

        # Setup KV caches AFTER model is on device, using the determined cache_dtype
        try:
            self.logger.debug(f"Setting up model KV caches (batch_size=1, dtype={cache_dtype})...")
            self._model.setup_caches(max_batch_size=1, dtype=cache_dtype)
            self.logger.debug("Model KV caches setup successfully.")
        except Exception as e:
             self.logger.error(f"Error setting up caches: {e}", exc_info=True)
             raise # Cache setup is critical

        # Load Text Tokenizer (Llama) - using the function defined above
        self.logger.debug("Loading text tokenizer (Llama)...")
        self._text_tokenizer = load_llama3_tokenizer(model_base_dir=model_base_dir)
        if self._text_tokenizer is None:
             raise RuntimeError("Failed to load Llama text tokenizer.")
        self.logger.debug("Text tokenizer loaded.")

        # Load Audio Tokenizer (Mimi) - using moshi loaders
        self.logger.debug("Loading audio tokenizer (Mimi)...")
        if loaders is None:
             raise ImportError("moshi.models.loaders could not be imported. Cannot load Mimi.")
        try:
            # Use a sub-cache dir within data_dir for HF downloads if needed
            hf_cache_dir = Path(data_dir()) / "models/hf_cache"
            hf_cache_dir.mkdir(parents=True, exist_ok=True)
            mimi_weight_path = hf_hub_download(
                repo_id=loaders.DEFAULT_REPO, # e.g., "collabora/whisperspeech"
                filename=loaders.MIMI_NAME, # e.g., "mimi.pth" ? Check actual filename from moshi loaders
                cache_dir=str(hf_cache_dir) # Use cache dir
            )
            # Load Mimi model, passing the target device
            mimi = loaders.get_mimi(mimi_weight_path, device=self.device)
            # Ensure it's on the correct device (might be redundant but safe)
            mimi = mimi.to(self.device)
            # Set the number of codebooks expected by the CSM model
            num_codebooks = model.config.audio_num_codebooks
            mimi.set_num_codebooks(num_codebooks) # Critical step from original repo
            self._audio_tokenizer = mimi
            self._num_codebooks = num_codebooks # Store for reference
            self.sample_rate = mimi.sample_rate # Get sample rate from Mimi
            self.logger.info(f"Audio tokenizer (Mimi) loaded successfully. Sample rate: {self.sample_rate}, Codebooks: {self._num_codebooks}")
        except Exception as e:
             self.logger.error(f"Failed to load or setup Mimi audio tokenizer: {e}", exc_info=True)
             self._audio_tokenizer = None
             self.sample_rate = 24000 # Fallback sample rate? Or raise error?
             self._num_codebooks = model.config.audio_num_codebooks
             raise RuntimeError("Failed to initialize Mimi audio tokenizer.") from e

        # Load Watermarker (using imported functions from watermarking.py)
        self.logger.debug("Loading watermarker...")
        if watermarking_available and load_watermarker is not None:
            try:
                # load_watermarker expects device string or torch.device
                self._watermarker = load_watermarker(device=self.device)
                self.logger.debug("Watermarker loaded successfully.")
            except Exception as e:
                 self.logger.error(f"Failed to load watermarker: {e}", exc_info=True)
                 self._watermarker = None
                 self.logger.warning("Watermarking will be disabled due to loading error.")
        else:
             self._watermarker = None
             self.logger.warning("Watermarking script/functions not available. Watermarking disabled.")

        # Store max sequence length from the model's backbone
        try:
            self.max_seq_len = model.backbone.max_seq_len
        except AttributeError:
            self.logger.warning("Could not get max_seq_len from model.backbone. Using default 2048.")
            self.max_seq_len = 2048 # Fallback

        self.logger.info(f"Generator initialization complete. Max sequence length: {self.max_seq_len}")

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes text using the Llama tokenizer and formats for the CSM model.
           Matches the logic in the original CSM repo's generator.py.
           Output shape: (seq_len, num_codebooks + 1)
        """
        frame_tokens_list = [] # Use list to append, then cat
        frame_masks_list = []

        if not text: # Handle empty text
             # Return empty tensors with the correct trailing dimension size
             return torch.empty((0, self._num_codebooks + 1), dtype=torch.long, device=self.device), \
                    torch.empty((0, self._num_codebooks + 1), dtype=torch.bool, device=self.device)

        try:
            # Encode text with speaker prefix, e.g., "[0]Hello there."
            # Original code uses f"[{speaker}]{text}" *before* encoding. Keep this.
            text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
            num_text_tokens = len(text_tokens)

            if num_text_tokens == 0: # Handle case where encoding results in nothing
                return torch.empty((0, self._num_codebooks + 1), dtype=torch.long, device=self.device), \
                       torch.empty((0, self._num_codebooks + 1), dtype=torch.bool, device=self.device)

            # Create frame tensor (zeros) for text tokens
            # Shape: (num_text_tokens, num_codebooks + 1)
            text_frame = torch.zeros(num_text_tokens, self._num_codebooks + 1, dtype=torch.long, device=self.device)
            text_frame_mask = torch.zeros(num_text_tokens, self._num_codebooks + 1, dtype=torch.bool, device=self.device)

            # Place text tokens in the *last column* (-1)
            text_frame[:, -1] = torch.tensor(text_tokens, device=self.device, dtype=torch.long)
            # Set mask for the text token column to True
            text_frame_mask[:, -1] = True

            # Append to lists (original repo concatenated directly, but list append is fine)
            frame_tokens_list.append(text_frame)
            frame_masks_list.append(text_frame_mask)

            # Concatenate the lists (will just be one item here)
            final_tokens = torch.cat(frame_tokens_list, dim=0)
            final_masks = torch.cat(frame_masks_list, dim=0)
            return final_tokens, final_masks

        except Exception as e:
            self.logger.error(f"Error tokenizing text segment: {e}", exc_info=True)
            return torch.empty((0, self._num_codebooks + 1), dtype=torch.long, device=self.device), \
                   torch.empty((0, self._num_codebooks + 1), dtype=torch.bool, device=self.device)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes audio using the Mimi tokenizer and formats for the CSM model.
           Matches the logic in the original CSM repo's generator.py.
           Output shape: (num_frames, num_codebooks + 1)
        """
        frame_tokens_list = [] # Use list to append, then cat
        frame_masks_list = []
        frame_size = self._num_codebooks + 1 # Total columns

        if self._audio_tokenizer is None:
            self.logger.error("Audio tokenizer (Mimi) not initialized.")
            return torch.empty((0, frame_size), dtype=torch.long, device=self.device), \
                   torch.empty((0, frame_size), dtype=torch.bool, device=self.device)

        if audio is None or audio.numel() == 0:
            self.logger.warning("Empty audio tensor passed to _tokenize_audio.")
            return torch.empty((0, frame_size), dtype=torch.long, device=self.device), \
                   torch.empty((0, frame_size), dtype=torch.bool, device=self.device)

        # --- Input Assertions (Moved BEFORE try block) ---
        assert audio.ndim == 1, f"Audio must be single channel (1D tensor), got shape {audio.shape}"
        # Ensure audio is on the correct device *before* passing to tokenizer
        if audio.device != self.device:
            self.logger.warning(f"Audio tensor for tokenization was on {audio.device}, moving to {self.device}.")
            audio = audio.to(self.device)
        # --- End Assertions ---

        try:
            # Encode audio: expects shape [B, 1, T], returns list of tensors [K, T_frames]
            # We use B=1, add channel dim, result is [0] access
            # Ensure input tensor has correct dimensions for Mimi encoder
            audio_codes = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0] # Shape [K, T_frames]

            # Ensure number of codebooks (K) matches model config
            if audio_codes.size(0) != self._num_codebooks:
                self.logger.warning(f"Mimi produced {audio_codes.size(0)} codebooks, model expects {self._num_codebooks}. Adjusting.")
                if audio_codes.size(0) > self._num_codebooks:
                    audio_codes = audio_codes[:self._num_codebooks, :]
                else: # Pad if fewer codebooks (less likely but possible)
                    padding = torch.zeros((self._num_codebooks - audio_codes.size(0), audio_codes.size(1)),
                                          dtype=audio_codes.dtype, device=self.device)
                    audio_codes = torch.cat([audio_codes, padding], dim=0)

            # Add EOS frame (column of zeros) - Convention from original CSM model
            eos_frame = torch.zeros(self._num_codebooks, 1, dtype=audio_codes.dtype, device=self.device)
            audio_codes_with_eos = torch.cat([audio_codes, eos_frame], dim=1) # Shape [K, T_frames + 1]

            # Transpose to [T_frames + 1, K] to align with model frame structure (time steps first)
            audio_tokens_t = audio_codes_with_eos.transpose(0, 1)
            num_frames = audio_tokens_t.size(0)

            # Create frame tensor (zeros) with shape (num_frames, num_codebooks + 1)
            audio_frame = torch.zeros(num_frames, frame_size, dtype=torch.long, device=self.device)
            audio_frame_mask = torch.zeros(num_frames, frame_size, dtype=torch.bool, device=self.device)

            # Place audio tokens (shape [T_frames+1, K]) into the first 'num_codebooks' columns
            audio_frame[:, :-1] = audio_tokens_t
            # Set mask for audio token columns to True
            audio_frame_mask[:, :-1] = True

            # Append to lists
            frame_tokens_list.append(audio_frame)
            frame_masks_list.append(audio_frame_mask)

            # Concatenate lists (will be one item here)
            final_tokens = torch.cat(frame_tokens_list, dim=0)
            final_masks = torch.cat(frame_masks_list, dim=0)
            return final_tokens, final_masks

        except Exception as e:
            self.logger.error(f"Error during _tokenize_audio: {e}", exc_info=True)
            return torch.empty((0, frame_size), dtype=torch.long, device=self.device), \
                   torch.empty((0, frame_size), dtype=torch.bool, device=self.device)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes a full Segment (text and potentially audio) and concatenates the results.
           Matches the logic in the original CSM repo's generator.py.
           Returns: (seq_len, num_codebooks + 1), (seq_len, num_codebooks + 1)
        """
        # Tokenize text part
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)

        # Tokenize audio part if present
        if segment.audio is not None and segment.audio.numel() > 0:
            # Ensure audio is on the correct device before tokenizing
            audio_tokens, audio_masks = self._tokenize_audio(segment.audio.to(self.device))
        else:
            # Create empty tensors if no audio
            frame_size = self._num_codebooks + 1
            audio_tokens = torch.empty((0, frame_size), dtype=torch.long, device=self.device)
            audio_masks = torch.empty((0, frame_size), dtype=torch.bool, device=self.device)

        # Concatenate text and audio frames along the sequence dimension (dim=0)
        return torch.cat([text_tokens, audio_tokens], dim=0), \
               torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000, # Max length of the *generated* audio part
        temperature: float = 0.8, # Adjusted default based on previous experiments? Original was 0.9
        topk: int = 50,
    ) -> torch.Tensor:
        """
        Generates audio waveform based on text, speaker, and context.
        Closely follows the logic and structure of the original CSM repo's generator.py.
        Returns the generated audio waveform as a 1D CPU tensor.
        """
        start_time_generate = time.perf_counter()
        self.logger.info(f"Starting generation for speaker {speaker}: '{text[:50]}...' (Context segments: {len(context)})")

        # --- 1. Reset KV Caches ---
        # Crucial for starting generation for a new sequence clean
        self._model.reset_caches()
        self.logger.debug("Model KV caches reset.")

        # --- 2. Calculate Max Generation Frames ---
        # Based on 1 frame = 80ms assumption (needs verification based on model training)
        # This limits the duration of the audio generated *in this call*, not the total sequence length.
        max_generation_len_frames = int(max_audio_length_ms / 80) # Example: 10000ms -> 125 frames
        self.logger.debug(f"Max generation frames set to: {max_generation_len_frames} (for {max_audio_length_ms}ms duration)")

        # --- 3. Tokenize Context and Current Text ---
        all_tokens_list: List[torch.Tensor] = []
        all_masks_list: List[torch.Tensor] = []
        try:
            # Tokenize historical context segments
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                all_tokens_list.append(segment_tokens)
                all_masks_list.append(segment_tokens_mask)

            # Tokenize the new text to be generated (this will be the prompt for audio generation)
            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
            all_tokens_list.append(gen_segment_tokens)
            all_masks_list.append(gen_segment_tokens_mask)
            self.logger.debug("Tokenization of context and current text complete.")
        except Exception as e:
            self.logger.error(f"Error during tokenization phase: {e}", exc_info=True)
            return torch.tensor([], device='cpu') # Return empty CPU tensor on error

        # --- 4. Concatenate Tokens into Prompt ---
        if not all_tokens_list:
             self.logger.warning("No tokens generated from context or input text. Cannot generate.")
             return torch.tensor([], device='cpu')

        prompt_tokens = torch.cat(all_tokens_list, dim=0)
        prompt_tokens_mask = torch.cat(all_masks_list, dim=0)
        prompt_len = prompt_tokens.size(0)
        self.logger.debug(f"Total prompt length (context + current text): {prompt_len} tokens.")

        # --- 5. Check Prompt Length Against Model Limits ---
        # Ensure prompt length doesn't exceed capacity when considering future generated frames.
        # max_seq_len is the absolute limit of the backbone transformer.
        max_context_len = self.max_seq_len - max_generation_len_frames
        if prompt_len >= self.max_seq_len:
             # If prompt *already* exceeds max length, truncate immediately.
             self.logger.warning(f"Initial prompt length ({prompt_len}) exceeds model max sequence length ({self.max_seq_len}). Truncating prompt.")
             prompt_tokens = prompt_tokens[-self.max_seq_len:]
             prompt_tokens_mask = prompt_tokens_mask[-self.max_seq_len:]
             prompt_len = prompt_tokens.size(0) # Update length after truncation
             self.logger.warning(f"Truncated prompt length to {prompt_len} tokens.")
        elif prompt_len >= max_context_len:
             # If prompt is okay now, but will exceed limit *with* generation, warn.
             # Original code raised ValueError here. Let's just warn.
             self.logger.warning(
                 f"Prompt length ({prompt_len}) plus max generation frames ({max_generation_len_frames}) may exceed max sequence length ({self.max_seq_len}). "
                 f"Effective max context was {max_context_len}. Generation might stop early if limit is hit."
             )
             # No truncation needed here, but generation loop needs length check.

        if prompt_len == 0:
             self.logger.error("Prompt length is zero after tokenization/truncation. Cannot generate.")
             return torch.tensor([], device='cpu')

        # --- 6. Prepare Initial Inputs for Generation Loop ---
        # Add batch dimension (B=1)
        curr_tokens_batched = prompt_tokens.unsqueeze(0) # Shape: [1, prompt_len, F]
        curr_masks_batched = prompt_tokens_mask.unsqueeze(0) # Shape: [1, prompt_len, F]
        # Input positions correspond to the token indices in the prompt
        curr_pos_batched = torch.arange(0, prompt_len, device=self.device).unsqueeze(0).long() # Shape: [1, prompt_len]

        # --- 7. Autoregressive Generation Loop ---
        generated_samples: List[torch.Tensor] = [] # Stores generated audio frames [1, K]
        self.logger.debug(f"Starting autoregressive generation loop (max_len={max_generation_len_frames} frames)...")
        log_interval = 50 # Log progress frequency
        start_loop_time = time.perf_counter()
        # Original CSM didn't have explicit min frames, let's omit for now or set low (e.g., 1)
        # min_frames_before_eos = 1

        for i in range(max_generation_len_frames):
            # Calculate the position index for the token *about* to be generated
            current_total_seq_len = curr_pos_batched.max().item() + 1
            if current_total_seq_len > self.max_seq_len:
                 self.logger.warning(f"Stopping generation at frame {i}: Next token position ({current_total_seq_len}) would exceed max sequence length ({self.max_seq_len}).")
                 break

            try:
                # Generate one frame (all codebooks) using the *current* state
                # Input shapes: tokens[1, S, F], mask[1, S, F], pos[1, S] -> S is current sequence length
                # Output shape: sample[1, K] (K = num_codebooks)
                sample = self._model.generate_frame(
                    curr_tokens_batched, curr_masks_batched, curr_pos_batched,
                    temperature, topk
                )

                # --- Check for End-of-Sequence (EOS) ---
                # Original CSM checks if the *entire frame* is zeros.
                is_eos = torch.all(sample == 0).item()
                # Add check: Only break on EOS if *some* frames have been generated (avoids immediate stop)
                if i > 0 and is_eos:
                    self.logger.info(f"EOS detected at frame {i+1}. Stopping generation.")
                    break
                # --- End EOS Check ---

                generated_samples.append(sample) # Store the generated frame [1, K]

                # --- Prepare input for the *next* iteration ---
                # The input for the next step is just the frame we just generated.
                # We need to format it correctly: shape [1, 1, F] (F = K+1 = num_codebooks+1)
                next_input_frame = torch.zeros(1, 1, self._num_codebooks + 1, dtype=torch.long, device=self.device)
                next_input_mask = torch.zeros(1, 1, self._num_codebooks + 1, dtype=torch.bool, device=self.device)

                # Place the generated audio sample (shape [1, K]) into the audio columns [:, :, :-1]
                next_input_frame[0, 0, :-1] = sample[0] # sample is [1, K]
                # Set the mask for the audio columns to True
                next_input_mask[0, 0, :-1] = True

                # Update state for the next step: inputs are just the new frame, position increments
                curr_tokens_batched = next_input_frame # Shape [1, 1, F]
                curr_masks_batched = next_input_mask   # Shape [1, 1, F]
                curr_pos_batched = curr_pos_batched[:, -1:] + 1 # Shape [1, 1], value is the next position index

                # Log progress
                if (i + 1) % log_interval == 0:
                     elapsed = time.perf_counter() - start_loop_time
                     self.logger.debug(f"Generated frame {i+1}/{max_generation_len_frames} ({elapsed:.2f}s)")

            except Exception as e:
                self.logger.error(f"Error during frame generation {i+1}: {e}", exc_info=True)
                # Log device states if it's a mismatch error
                if "Expected all tensors to be on the same device" in str(e):
                    self.logger.error(f"Device state: Model={next(self._model.parameters()).device}, "
                                     f"Tokens={curr_tokens_batched.device}, Mask={curr_masks_batched.device}, Pos={curr_pos_batched.device}")
                break # Stop generation on error

        loop_duration = time.perf_counter() - start_loop_time
        self.logger.info(f"Generation loop finished in {loop_duration:.2f} seconds. Generated {len(generated_samples)} frames.")

        # --- 8. Decode Generated Audio Frames ---
        if not generated_samples:
            self.logger.warning("No audio frames were generated.")
            return torch.tensor([], device='cpu')

        if self._audio_tokenizer is None:
             self.logger.error("Audio tokenizer not available for decoding.")
             return torch.tensor([], device='cpu')

        self.logger.debug("Decoding generated audio frames using Mimi...")
        try:
            # Stack the list of [1, K] tensors -> [N, 1, K]
            stacked_samples_n1k = torch.stack(generated_samples)
            # Permute to match decoder expected shape [B, K, T_frames] -> [1, K, N]
            stacked_samples_1kn = stacked_samples_n1k.permute(1, 2, 0)

            # Decode expects shape [B, K, T_frames]
            # Decoding might be faster on CPU, but original repo implies device consistency. Let's keep on device for now.
            decoded_audio = self._audio_tokenizer.decode(stacked_samples_1kn) # Shape: [1, 1, T_samples]
            # Squeeze batch and channel dimensions -> [T_samples]
            decoded_audio = decoded_audio.squeeze(0).squeeze(0)
            self.logger.debug(f"Audio decoded successfully. Shape: {decoded_audio.shape}")
        except Exception as e:
            self.logger.error(f"Error decoding audio frames: {e}", exc_info=True)
            return torch.tensor([], device='cpu')

        # --- 9. Apply Watermarking ---
        if self._watermarker is not None and watermark is not None and CSM_1B_GH_WATERMARK is not None:
            self.logger.debug("Applying watermark...")
            try:
                # Ensure audio is on the correct device for the watermarker
                audio_to_watermark = decoded_audio.to(self.device)
                # Use the public watermark key from the original repo
                # watermark() returns (watermarked_audio, watermark_sample_rate)
                watermarked_audio, wm_sample_rate = watermark(
                    self._watermarker, audio_to_watermark, self.sample_rate, CSM_1B_GH_WATERMARK
                )
                # Resample back to the original sample rate if watermarking changed it
                if wm_sample_rate != self.sample_rate:
                    self.logger.debug(f"Resampling watermarked audio from {wm_sample_rate}Hz to {self.sample_rate}Hz")
                    final_audio = torchaudio.functional.resample(
                        watermarked_audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate
                    )
                else:
                    final_audio = watermarked_audio
                self.logger.debug("Watermark applied successfully.")
            except Exception as e:
                self.logger.error(f"Failed to apply watermark: {e}", exc_info=True)
                final_audio = decoded_audio # Use unwatermarked audio on error
        else:
            self.logger.debug("Watermarker not available or key missing, skipping watermarking.")
            final_audio = decoded_audio
        # --- End Watermarking ---

        # --- 10. Return Final Audio Tensor ---
        total_time = time.perf_counter() - start_time_generate
        self.logger.info(f"Generator.generate completed in {total_time:.2f} seconds.")
        # Return final audio tensor on CPU as expected by torchaudio.save etc.
        return final_audio.cpu()

# --- Loading function (Keep from previous step) ---
def load_csm_1b_local(model_path: str, device: str = "cuda") -> Generator:
    """
    Loads the CSM-1B model from a local base path and initializes the Generator.
    Expects subdirectories like 'csm-1b' and 'Llama-3.2-1B' within model_path.
    """
    logger.info(f"--- Loading CSM-1B Locally ---")
    target_device = torch.device(device)
    logger.info(f"Target device: {target_device}")
    base_model_path = Path(model_path)
    csm_specific_path = base_model_path / "csm-1b" # Expected subdir for CSM weights/config

    logger.info(f"Attempting to load CSM model weights from: {csm_specific_path}")
    if not csm_specific_path.is_dir():
         logger.error(f"CSM local model directory not found: {csm_specific_path}")
         raise FileNotFoundError(f"CSM local model directory not found: {csm_specific_path}")

    try:
        # Load model from local path (weights loaded to CPU by from_local_pretrained)
        # Ensure Model.from_local_pretrained is defined in the updated models.py
        model = Model.from_local_pretrained(str(csm_specific_path))

        # Determine target dtype for model parameters based on device
        # Use bfloat16 on CUDA if available, matching original CSM preference
        if target_device.type == 'cuda':
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info(f"Targeting {target_dtype} for model parameters on CUDA.")
        else:
            target_dtype = torch.float32
            logger.info(f"Targeting {target_dtype} for model parameters on {target_device.type}.")

        # Move model to target device AND cast its parameters
        model = model.to(device=target_device, dtype=target_dtype)
        model.eval() # Set to evaluation mode

        # Log parameter info after move/cast
        log_dtype = next(model.parameters()).dtype if list(model.parameters()) else 'N/A'
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded, moved to {target_device} (dtype: {log_dtype}), params: {num_params/1e9:.2f}B, set to eval mode.")

    except Exception as e:
        logger.error(f"Failed during local model loading or device/dtype transfer: {e}", exc_info=True)
        raise

    # Initialize the Generator, passing the loaded model, device, and base model path (for tokenizer loading)
    logger.info("CSM Model ready. Creating Generator instance...")
    try:
        # Pass the target_device object and the base_model_path
        generator = Generator(model, device=target_device, model_base_dir=base_model_path)
        return generator
    except Exception as e:
         logger.error(f"Failed to initialize Generator after loading model: {e}", exc_info=True)
         raise