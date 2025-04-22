# ./python-app/src/sesame_tts.py
import os
import torch
import torchaudio
import asyncio
import tempfile
import subprocess
from datetime import datetime
import logging
from pathlib import Path
import time # Use time module for perf_counter
from typing import Tuple
from env import models_dir as models_dir_env, data_dir

# --- Use the local loader ---
from generator import load_csm_1b_local

logger = logging.getLogger(__name__)

# --- Function to convert WAV to MP3 (can be kept here or moved to utils) ---
def convert_to_mp3(wav_file_path: Path, mp3_file_path: Path) -> Tuple[bool, float]:
    """Converts a WAV file to MP3 using ffmpeg, returns success and duration."""
    conversion_time_ms = -1.0
    try:
        logger.info(f"Converting '{wav_file_path.name}' to '{mp3_file_path.name}'...")
        start_conv = time.perf_counter()
        subprocess.run(
            ["ffmpeg", "-i", str(wav_file_path), "-qscale:a", "2", str(mp3_file_path), "-y", "-loglevel", "error"],
            check=True, capture_output=True
        )
        end_conv = time.perf_counter()
        conversion_time_ms = (end_conv - start_conv) * 1000
        logger.info(f"Successfully converted '{wav_file_path.name}' to '{mp3_file_path.name}' in {conversion_time_ms:.2f} ms")
        return True, conversion_time_ms
    except FileNotFoundError:
        logger.error("ERROR: ffmpeg command not found. Please install ffmpeg.")
        return False, conversion_time_ms
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during ffmpeg conversion for {wav_file_path.name}:")
        logger.error(f"Stderr: {e.stderr.decode()}")
        return False, conversion_time_ms
    except Exception as e:
        logger.error(f"Error converting {wav_file_path.name} to MP3: {e}", exc_info=True)
        return False, conversion_time_ms

class SesameTTS:
    def __init__(self, device="cpu", model_dir=None):
        logger.info(f"Initializing SesameTTS class with device={device}, model_dir={model_dir}")
        # os.environ["NO_TORCH_COMPILE"] = "1" # May not be needed if generator doesn't compile

        # --- Resolve model directory (relative to src/ where main.py/app.py run) ---
        resolved_model_dir = Path(models_dir_env())
        logger.info(f"SesameTTS resolved model directory to: {resolved_model_dir}")
        # --- End Resolve ---

        self.generator = None
        self.sample_rate = 24000 # Default fallback
        self.tts_available = False
        self.device = device # Store the device string

        try:
            # Set HF env vars (might be redundant if set early in main.py, but safe)
            # os.environ["TRANSFORMERS_OFFLINE"] = "1" # Optional
            os.environ["HF_HOME"] = str(resolved_model_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(resolved_model_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(resolved_model_dir)

            logger.info(f"Loading CSM model via load_csm_1b_local (Path: {resolved_model_dir}, Device: {self.device})...")
            # --- Use the local loader ---
            self.generator = load_csm_1b_local(model_path=str(resolved_model_dir), device=self.device)
            # --- End Use ---

            if not hasattr(self.generator, 'generate') or not hasattr(self.generator, 'sample_rate'):
                 raise TypeError("Loaded generator object does not have expected 'generate' method or 'sample_rate' attribute.")

            self.sample_rate = self.generator.sample_rate
            logger.info(f"SesameTTS initialized successfully. Generator loaded. Sample Rate: {self.sample_rate}")
            self.tts_available = True

        except Exception as e:
            logger.error(f"Failed to initialize SesameTTS generator: {str(e)}", exc_info=True)
            # Ensure attributes are set for graceful failure
            self.generator = None
            self.sample_rate = 24000
            self.tts_available = False
            logger.warning(f"Using fallback (silent) audio generation due to initialization error.")

    async def generate_audio_and_convert(self, text, speaker_id=0, output_dir="static/audio"):
        """Generates audio, saves as temp WAV, converts to MP3, deletes WAV, returns MP3 path."""
        if not self.tts_available or self.generator is None:
            logger.warning("TTS not available/loaded, skipping audio generation.")
            return None

        temp_wav_file_path = None # Define variable outside try
        abs_output_dir = Path(data_dir()) / output_dir # Create absolute path for output
        abs_output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists

        try:
            start_time_gen = time.perf_counter()
            logger.info(f"TTS generating audio for speaker {speaker_id}: '{text[:30]}...'")

            # Call the generate method of the Generator instance
            audio_tensor = self.generator.generate(
                text=text,
                speaker=speaker_id,
                context=[], # Pass context if needed
                max_audio_length_ms=20_000, # Or adjust as needed
                temperature=0.8, # Default params
                topk=50
            )

            generation_time = (time.perf_counter() - start_time_gen) * 1000
            logger.info(f"Audio tensor generated in {generation_time:.2f} ms, shape={audio_tensor.shape if audio_tensor is not None else 'None'}")

            if audio_tensor is None or audio_tensor.numel() == 0:
                logger.error("Generation resulted in empty tensor.")
                return None

            # Create unique filename for MP3
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            base_filename = f"audio_spk{speaker_id}_{timestamp}"
            mp3_file = abs_output_dir / f"{base_filename}.mp3"

            # --- Use tempfile for WAV ---
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(abs_output_dir)) as f:
                temp_wav_file_path = Path(f.name)
            logger.debug(f"Saving temporary WAV to: {temp_wav_file_path.name}")

            # Save WAV (tensor must be on CPU)
            torchaudio.save(str(temp_wav_file_path), audio_tensor.cpu().unsqueeze(0), self.sample_rate)
            logger.debug(f"Temporary WAV saved successfully.")
            # --- End tempfile ---

            # Convert to MP3
            conversion_ok, conversion_ms = convert_to_mp3(temp_wav_file_path, mp3_file)

            if conversion_ok:
                 logger.info(f"MP3 conversion successful ({conversion_ms:.2f} ms).")
                 return str(mp3_file) # Return path to the MP3
            else:
                 logger.error("MP3 conversion failed.")
                 return None # Indicate failure

        except Exception as e:
            logger.error(f"Audio generation/conversion failed: {str(e)}", exc_info=True)
            return None
        finally:
            # --- Ensure temp WAV is deleted ---
            if temp_wav_file_path is not None and temp_wav_file_path.exists():
                 try:
                     os.remove(temp_wav_file_path)
                     logger.debug(f"Temporary WAV file deleted: {temp_wav_file_path.name}")
                 except OSError as e:
                     logger.warning(f"Could not delete temporary WAV file {temp_wav_file_path.name}: {e}")
            # --- End cleanup ---

# Example usage (if run directly - careful with paths if not run from src/)
if __name__ == "__main__":
    import asyncio
    async def main():
         print("Running SesameTTS direct test...")
         # Assume models are in ../models relative to this script's location (src/)
         model_path = Path(models_dir_env())
         print(f"Looking for models in: {model_path}")
         tts = SesameTTS(device="cpu", model_dir=str(model_path))
         if tts.tts_available:
              print("TTS Initialized. Generating test audio...")
              mp3_path = await tts.generate_audio_and_convert("Hello, how are you?", speaker_id=0)
              if mp3_path:
                   print(f"Test audio generated successfully: {mp3_path}")
              else:
                   print("Test audio generation failed.")
         else:
              print("TTS failed to initialize.")

    asyncio.run(main())