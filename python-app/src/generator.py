# ./python-app/src/generator.py
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
from models import Model, ModelArgs, llama3_2_1B
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
import os
import json
from pathlib import Path
import logging
from subprocess import run
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import time
from env import models_dir

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)

try:
    from moshi.models import loaders
except ImportError:
    logger.error("Failed to import 'loaders' from 'moshi.models'. Please ensure 'moshi-tts' is installed.")
    loaders = None

@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor

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
            self._model.to(self.device) # Ensure model is on device before setup_caches
            # Call setup_caches without dtype - it will determine internally based on model parameters
            self._model.setup_caches(max_batch_size=1)
            logger.debug(f"Model caches setup successfully during init (dtype determined internally).")

            # --- LAST DITCH: Try forcing cache_pos device AGAIN *after* setup_caches ---
            logger.debug(f"Attempting POST-SETUP correction of KVCache.cache_pos tensors to: {device}...")
            corrected_count_post = 0
            target_cache_pos_dtype = torch.long # KVCache usually uses long for positions
            for name, module in self._model.named_modules():
                if module.__class__.__name__ == "KVCache":
                    if hasattr(module, 'cache_pos') and isinstance(module.cache_pos, torch.Tensor):
                        current_cache_pos = module.cache_pos
                        if current_cache_pos.device != device:
                            logger.warning(f"POST-SETUP: KVCache module '{name}' has cache_pos on incorrect device ({current_cache_pos.device}). Forcing replacement.")
                            try:
                                max_len = current_cache_pos.shape[0]
                                # Delete the existing buffer attribute
                                del module.cache_pos
                                if "cache_pos" in module._buffers:
                                    del module._buffers["cache_pos"]

                                # Create the new tensor on the target device with expected dtype
                                new_cache_pos = torch.arange(max_len, device=device, dtype=target_cache_pos_dtype)

                                # Register the new tensor as a buffer
                                module.register_buffer("cache_pos", new_cache_pos, persistent=False)
                                logger.info(f"POST-SETUP: Successfully replaced cache_pos buffer for KVCache module '{name}' with tensor on device {device} (dtype={target_cache_pos_dtype}).")
                                corrected_count_post += 1
                            except Exception as e_buf_post:
                                logger.error(f"POST-SETUP: Failed to replace cache_pos buffer for KVCache module '{name}': {e_buf_post}", exc_info=True)
                        # else:
                        #     logger.debug(f"POST-SETUP: KVCache module '{name}' cache_pos already on correct device ({device}).")
                    # elif hasattr(module, 'cache_pos'):
                    #     logger.warning(f"POST-SETUP: KVCache module '{name}' has cache_pos, but it's not a Tensor? Type: {type(module.cache_pos)}")

            logger.debug(f"Finished POST-SETUP KVCache.cache_pos device correction check. Replaced {corrected_count_post} buffers.")
            # --- END LAST DITCH ---

        except Exception as e:
             logger.error(f"Error during Generator init (including cache setup/correction): {e}", exc_info=True)
             raise # Re-raise if setup fails

        logger.debug("Loading text tokenizer (Llama)...")
        self._text_tokenizer = self.load_llama3_tokenizer()
        logger.debug("Text tokenizer loaded.")

        logger.debug(f"Loading Mimi model and feature extractor (kyutai/mimi) to device: {self.device}...")
        try:
            if loaders is None:
                raise ImportError("moshi.models.loaders could not be imported.")
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

    def _ensure_cache_pos_on_device(self):
        """Ensure that cache_pos in KVCache is on the correct device"""
        for name, module in self._model.named_modules():
            if hasattr(module, 'cache_pos') and isinstance(module.cache_pos, torch.Tensor):
                if module.cache_pos.device != self.device:
                    logger.warning(f"cache_pos in {name} is on {module.cache_pos.device}, moving to {self.device}")
                    new_cache_pos = module.cache_pos.to(self.device)
                    del module._buffers['cache_pos']
                    module.register_buffer('cache_pos', new_cache_pos)
                logger.debug(f"cache_pos in {name} is on {module.cache_pos.device}")

    def load_llama3_tokenizer(self):
        local_tokenizer_path = Path(models_dir()) / "llama-3-2-1b"
        hub_identifier = "unsloth/Llama-3.2-1B"

        tokenizer = None
        if local_tokenizer_path.is_dir():
            try:
                logger.info(f"Attempting to load Llama tokenizer from local path: {local_tokenizer_path}")
                tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer_path))
                logger.info("Successfully loaded Llama tokenizer from local path.")
            except Exception as e:
                logger.warning(f"Failed to load Llama tokenizer from local path {local_tokenizer_path}: {e}")
        else:
            logger.info(f"Local Llama tokenizer path not found: {local_tokenizer_path}. Attempting Hub download.")

        if tokenizer is None:
            try:
                logger.info(f"Attempting to load Llama tokenizer from Hub: {hub_identifier}")
                tokenizer = AutoTokenizer.from_pretrained(hub_identifier)
                logger.info(f"Successfully loaded Llama tokenizer from Hub/cache ({hub_identifier}).")
            except Exception as e:
                logger.error(f"CRITICAL: Failed to load Llama tokenizer from both local path and Hub: {e}", exc_info=True)
                raise

        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        if bos and eos:
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
            )
        else:
            logger.warning("BOS or EOS token not found in tokenizer, cannot set TemplateProcessing.")
        return tokenizer

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not text:
            return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        if not text_tokens:
            return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)
        text_frame = torch.zeros(len(text_tokens), 33, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(len(text_tokens), 33, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True
        return text_frame, text_frame_mask

    @torch.inference_mode()
    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, '_audio_tokenizer') or self._audio_tokenizer is None:
            logger.error("Audio tokenizer (mimi) not initialized.")
            return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)
        if audio.numel() == 0:
            logger.warning("Empty audio tensor passed to _tokenize_audio.")
            return torch.empty((0, 33), dtype=torch.long, device=self.device), torch.empty((0, 33), dtype=torch.bool, device=self.device)
        assert audio.ndim == 1, f"Audio must be single channel (1D tensor), got shape {audio.shape}"

        audio_codes = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        num_codebooks = self._model.config.audio_num_codebooks
        if audio_codes.size(0) != num_codebooks:
            logger.warning(f"Mimi produced {audio_codes.size(0)} codebooks, but model expects {num_codebooks}. Adjusting.")
            audio_codes = audio_codes[:num_codebooks, :] if audio_codes.size(0) > num_codebooks else audio_codes

        eos_frame = torch.zeros(num_codebooks, 1, device=self.device, dtype=audio_codes.dtype)
        audio_codes_with_eos = torch.cat([audio_codes, eos_frame], dim=1)
        audio_tokens_t = audio_codes_with_eos.transpose(0, 1)
        num_frames = audio_tokens_t.size(0)

        frame_size = num_codebooks + 1
        audio_frame = torch.zeros(num_frames, frame_size, dtype=torch.long, device=self.device)
        audio_frame_mask = torch.zeros(num_frames, frame_size, dtype=torch.bool, device=self.device)
        audio_frame[:, :-1] = audio_tokens_t
        audio_frame_mask[:, :-1] = True
        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        if segment.audio is not None and segment.audio.numel() > 0:
            audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        else:
            audio_tokens = torch.empty((0, 33), dtype=torch.long, device=self.device)
            audio_masks = torch.empty((0, 33), dtype=torch.bool, device=self.device)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(self, text: str, speaker: int, context: List[Segment], max_audio_length_ms: float = 90_000, temperature: float = 0.8, topk: int = 50):
        start_time_generate = time.perf_counter()
        target_device = self.device
        logger.debug(f"Entering Generator.generate for speaker {speaker} on device {target_device} text: {text[:30]}...")

        if target_device.type == "cuda":
            torch.cuda.empty_cache()

        max_generation_len = int(max_audio_length_ms / 80)
        max_seq_len = 2048

        tokens_segments = [self._tokenize_segment(s) for s in context]
        tokens_segments.append(self._tokenize_text_segment(text, speaker))

        prompt_tokens = torch.empty((0, self._model.config.audio_num_codebooks + 1), dtype=torch.long, device=target_device)
        prompt_tokens_mask = torch.empty((0, self._model.config.audio_num_codebooks + 1), dtype=torch.bool, device=target_device)
        if tokens_segments:
            all_segment_tokens = [seg[0] for seg in tokens_segments]
            all_segment_masks = [seg[1] for seg in tokens_segments]
            prompt_tokens = torch.cat(all_segment_tokens, dim=0)
            prompt_tokens_mask = torch.cat(all_segment_masks, dim=0)

        bos_frame = torch.zeros(1, self._model.config.audio_num_codebooks + 1, dtype=torch.long, device=target_device)
        bos_frame_mask = torch.zeros(1, self._model.config.audio_num_codebooks + 1, dtype=torch.bool, device=target_device)
        if self._text_tokenizer and hasattr(self._text_tokenizer, 'bos_token_id') and self._text_tokenizer.bos_token_id is not None:
            bos_id = self._text_tokenizer.bos_token_id
            bos_frame[0, -1] = bos_id
            bos_frame_mask[0, -1] = True
            prompt_tokens = torch.cat((bos_frame, prompt_tokens), dim=0)
            prompt_tokens_mask = torch.cat((bos_frame_mask, prompt_tokens_mask), dim=0)

        if prompt_tokens.size(0) > max_seq_len:
            logger.warning(f"Prompt length ({prompt_tokens.size(0)}) exceeds max_seq_len ({max_seq_len}). Truncating.")
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0).to(target_device)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0).to(target_device)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=target_device).unsqueeze(0).long()

        MIN_FRAMES_BEFORE_TERMINATION = 25
        for i in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if i >= MIN_FRAMES_BEFORE_TERMINATION and torch.all(sample == 0):
                logger.info(f"Termination condition detected at frame {i+1} (after min threshold).")
                break
            samples.append(sample)

            next_token_frame = torch.zeros(1, 1, self._model.config.audio_num_codebooks + 1, dtype=torch.long, device=target_device)
            next_token_frame_mask = torch.zeros(1, 1, self._model.config.audio_num_codebooks + 1, dtype=torch.bool, device=target_device)
            next_token_frame[0, 0, :self._model.config.audio_num_codebooks] = sample[0]
            next_token_frame_mask[0, 0, :self._model.config.audio_num_codebooks] = True
            curr_tokens = next_token_frame
            curr_tokens_mask = next_token_frame_mask
            curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            logger.warning("No audio frames were generated.")
            return torch.tensor([])

        stacked_samples = torch.stack(samples).permute(1, 2, 0)
        decoded_audio = self._audio_tokenizer.decode(stacked_samples.cpu())
        decoded_audio = decoded_audio.squeeze(0).squeeze(0)
        logger.info(f"Generator.generate completed in {time.perf_counter() - start_time_generate:.2f} seconds total.")
        return decoded_audio.cpu()

def load_csm_1b_local(model_path: str, device: str = "cuda", audio_num_codebooks: int = 32):
    target_device = torch.device(device)
    csm_specific_path = Path(model_path) / "csm-1b"
    logger.info(f"Attempting to load CSM-1B model from local directory: {csm_specific_path}")

    if not csm_specific_path.is_dir():
        raise FileNotFoundError(f"CSM local model directory not found: {csm_specific_path}")

    model = Model.from_local_pretrained(str(csm_specific_path))
    target_dtype = torch.float32
    logger.warning(f"FORCING FLOAT32 dtype ({target_device.type}).")
    model = model.to(device=target_device, dtype=target_dtype)
    model.eval()
    logger.info(f"Model moved to {target_device} and set to eval mode.")
    return Generator(model, device=target_device)

def load_csm_1b(device: str = "cuda", model_dir: Path | str | None = None) -> Generator:
    target_device = torch.device(device)
    logger.info(f"Attempting to load CSM-1B model from Hub/cache onto device: {target_device}")
    model = Model.from_pretrained("sesame/csm-1b", cache_dir=str(model_dir) if model_dir else None)
    model.eval()
    model = model.to(target_device)
    logger.info("Model loaded successfully. Creating generator...")
    return Generator(model, device=target_device)