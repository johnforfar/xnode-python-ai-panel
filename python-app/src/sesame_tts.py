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
from generator import load_csm_1b

logger = logging.getLogger(__name__)

# Add the conversion function
def convert_to_mp3(wav_file, mp3_file):
    try:
        print(f"Converting {wav_file} to {mp3_file}...")
        # Ensure ffmpeg is installed and in PATH
        subprocess.run(["ffmpeg", "-i", wav_file, "-qscale:a", "2", mp3_file, "-y", "-loglevel", "error"], check=True)
        print(f"Successfully converted {wav_file} to {mp3_file}")
        return True
    except FileNotFoundError:
        print("ERROR: ffmpeg command not found. Please install ffmpeg.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion: {e}")
        return False
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return False

class SesameTTS:
    def __init__(self, device="cpu", model_dir=None):
        logger.info(f"Initializing SesameTTS with device={device}, model_dir={model_dir}")
        os.environ["NO_TORCH_COMPILE"] = "1"

        # Resolve model directory
        if model_dir is None or not os.path.isabs(model_dir):
            try:
                from models import PROJECT_ROOT
                resolved_model_dir = PROJECT_ROOT / (model_dir or "models")
                logger.info(f"Resolved model directory to: {resolved_model_dir}")
            except ImportError:
                 logger.error("Could not import PROJECT_ROOT from models.py. Using relative path.")
                 resolved_model_dir = Path(model_dir or "models")
        else:
            resolved_model_dir = Path(model_dir)

        try:
            # Set env vars if needed
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HOME"] = str(resolved_model_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(resolved_model_dir)

            logger.info(f"Loading CSM model via load_csm_1b from generator.py (Path: {resolved_model_dir})...")
            # Call the load_csm_1b function which now handles local loading
            # Pass the resolved Path object
            self.generator = load_csm_1b(device=device, model_dir=resolved_model_dir)

            if not hasattr(self.generator, 'generate') or not hasattr(self.generator, 'sample_rate'):
                 raise TypeError("Loaded generator object does not have expected 'generate' method or 'sample_rate' attribute.")

            self.sample_rate = self.generator.sample_rate
            logger.info(f"SesameTTS initialized successfully. Generator loaded. Sample Rate: {self.sample_rate}")
            self.tts_available = True

        except Exception as e:
            logger.error(f"Failed to initialize SesameTTS: {str(e)}", exc_info=True)
            self.generator = None
            self.sample_rate = 24000
            self.tts_available = False
            logger.info(f"Using fallback (silent) audio generator due to initialization error.")
    
    async def generate_audio_and_convert(self, text, speaker_id=0, output_dir="static/audio"):
        """Generates audio, saves as WAV, converts to MP3, returns MP3 path."""
        if not self.tts_available or self.generator is None:
            logger.warning("TTS not available/loaded, skipping audio generation.")
            return None

        try:
            start_time = asyncio.get_event_loop().time()
            logger.info(f"TTS generating audio for speaker {speaker_id}: '{text[:30]}...'")

            # Call the generate method of the Generator instance
            audio_tensor = self.generator.generate(
                text=text,
                speaker=speaker_id,
                context=[], # Pass context if available/needed
                max_audio_length_ms=20_000,
            ) # Output should be on CPU already based on Generator.generate

            generation_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info(f"Audio tensor generated in {generation_time:.2f} ms, shape={audio_tensor.shape}")

            # Create unique filename
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            base_filename = f"audio_spk{speaker_id}_{timestamp}"
            wav_file = os.path.join(output_dir, f"{base_filename}.wav")
            mp3_file = os.path.join(output_dir, f"{base_filename}.mp3")

            # Save WAV (tensor must be on CPU)
            torchaudio.save(wav_file, audio_tensor.cpu(), self.sample_rate)
            logger.info(f"Audio saved to {wav_file}")

            # Convert to MP3
            conversion_start_time = asyncio.get_event_loop().time()
            if convert_to_mp3(wav_file, mp3_file):
                conversion_time_ms = (asyncio.get_event_loop().time() - conversion_start_time) * 1000
                logger.info(f"MP3 conversion successful in {conversion_time_ms:.2f} ms.")
                # Optionally remove WAV
                try: os.remove(wav_file)
                except OSError as e: logger.warning(f"Could not remove temp WAV file: {e}")
                return mp3_file # Return path to the MP3
            else:
                logger.error("MP3 conversion failed.")
                return None # Indicate failure

        except Exception as e:
            logger.error(f"Audio generation/conversion failed: {str(e)}", exc_info=True)
            return None

# Example usage
if __name__ == "__main__":
    import asyncio
    # Force CPU usage explicitly
    tts = SesameTTS(device="cpu", model_dir="/models")
    audio = asyncio.run(tts.generate_audio_and_convert("Hello, how are you?", speaker_id=0))
    torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), tts.generator.sample_rate)