import os
import sys
import time
from pathlib import Path
import logging
import torch
import torchaudio
import glob
import subprocess
import tempfile
from typing import Tuple
from env import models_dir

# --- Setup Project Paths relative to THIS script ---
SCRIPT_DIR = Path(__file__).resolve().parent # Should be python-app/src/tests/
SRC_DIR = SCRIPT_DIR.parent                 # Should be python-app/src/
PROJECT_ROOT = SRC_DIR.parent.parent        # Should be project root
MODELS_DIR = Path(models_dir())
OUTPUT_DIR = SCRIPT_DIR                     # Save output WAV/MP3 in the same tests/ directory

print(f"INFO: Project Root: {PROJECT_ROOT}")
print(f"INFO: Source Dir: {SRC_DIR}")
print(f"INFO: Models Dir: {MODELS_DIR}")
print(f"INFO: Output Dir (for audio files): {OUTPUT_DIR}")

# --- Add src to Python path ---
sys.path.insert(0, str(SRC_DIR))

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                    handlers=[logging.StreamHandler()]) # Log directly to console
logger = logging.getLogger("MULTI_TTS_TEST") # Renamed logger
logging.getLogger("transformers").setLevel(logging.ERROR) # Suppress verbose logs
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torchtune").setLevel(logging.ERROR)

# --- Set Hugging Face Cache Environment Variables ---
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
logger.info(f"HF Cache variables set to target: {MODELS_DIR}")

# --- Import TTS components ---
try:
    # Import the function specifically designed for local loading
    from generator import load_csm_1b_local
except ImportError as e:
    logger.error(f"Failed to import 'load_csm_1b_local' from generator: {e}")
    logger.error(f"Ensure '{SRC_DIR}/generator.py' exists and is importable.")
    sys.exit(1)
except Exception as e:
     logger.error(f"An unexpected error occurred during imports: {e}", exc_info=True)
     sys.exit(1)

# --- Test Configuration ---
TEST_TEXT = "Sam is the Man with a Plan!"
# --- Updated: Test speakers 0 through 16 ---
MAX_SPEAKER_ID_TO_TEST = 16 # Test up to ID 16
# --- End Update ---
BASE_MODEL_PATH = str(MODELS_DIR)
EXPECTED_CSM_SUBDIR = "csm-1b"

# --- Function to save tensor temporarily and convert WAV to MP3 ---
def save_temp_wav_and_convert_mp3(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    mp3_file_path: Path,
    output_dir: Path = OUTPUT_DIR # Use global OUTPUT_DIR
) -> Tuple[bool, float]:
    """Saves audio tensor to a temporary WAV, converts to MP3, logs time, deletes temp WAV."""
    temp_wav_file = None
    conversion_time_ms = -1.0
    try:
        # 1. Create a temporary WAV file path
        # Using NamedTemporaryFile ensures it's cleaned up even if script crashes mid-way
        # Suffix=".wav" helps ffmpeg identify the format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(output_dir)) as f:
            temp_wav_file_path = Path(f.name)
        logger.debug(f"Creating temporary WAV: {temp_wav_file_path.name}")

        # 2. Save audio tensor to temporary WAV
        audio_to_save = audio_tensor.cpu()
        if audio_to_save.ndim == 1:
            audio_to_save = audio_to_save.unsqueeze(0)
        torchaudio.save(str(temp_wav_file_path), audio_to_save, sample_rate)
        logger.debug(f"Temporary WAV saved successfully.")

        # 3. Convert to MP3 and time it
        logger.info(f"Converting temporary WAV to '{mp3_file_path.name}'...")
        start_conv = time.perf_counter()
        process = subprocess.run(
            ["ffmpeg", "-i", str(temp_wav_file_path), "-qscale:a", "2", str(mp3_file_path), "-y", "-loglevel", "error"],
            check=True, capture_output=True
        )
        end_conv = time.perf_counter()
        conversion_time_ms = (end_conv - start_conv) * 1000
        logger.info(f"Successfully converted to '{mp3_file_path.name}' in {conversion_time_ms:.2f} ms")
        return True, conversion_time_ms

    except FileNotFoundError:
        logger.error("ERROR: ffmpeg command not found. Please install ffmpeg.")
        return False, conversion_time_ms
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during ffmpeg conversion for {mp3_file_path.name}:")
        logger.error(f"Stderr: {e.stderr.decode()}")
        return False, conversion_time_ms
    except Exception as e:
        logger.error(f"Error during temp WAV save or MP3 conversion for {mp3_file_path.name}: {e}", exc_info=True)
        return False, conversion_time_ms
    finally:
        # 4. Clean up temporary WAV file
        if temp_wav_file_path is not None and temp_wav_file_path.exists():
            try:
                os.remove(temp_wav_file_path)
                logger.debug(f"Temporary WAV file deleted: {temp_wav_file_path.name}")
            except OSError as e:
                logger.warning(f"Could not delete temporary WAV file {temp_wav_file_path.name}: {e}")

# --- Main Test Function ---
def run_multi_tts_test():
    """Runs the TTS generation test for multiple speakers, stopping on error."""
    logger.info("--- Starting Multi-Speaker TTS Generation Test ---")

    # --- Delete existing WAV and MP3 files ---
    logger.info(f"Cleaning up existing .wav and .mp3 files in {OUTPUT_DIR}...")
    deleted_count = 0
    for file_pattern in ["*.wav", "*.mp3"]:
        for file_path in glob.glob(str(OUTPUT_DIR / file_pattern)):
            try:
                os.remove(file_path)
                logger.info(f"  Deleted: {os.path.basename(file_path)}")
                deleted_count += 1
            except OSError as e:
                logger.error(f"  Error deleting {file_path}: {e}")
    logger.info(f"Cleanup complete. Deleted {deleted_count} file(s).")

    # 1. Check for expected local model directory
    csm_model_dir = MODELS_DIR / EXPECTED_CSM_SUBDIR
    if not csm_model_dir.is_dir():
        logger.error(f"Required local model directory not found: {csm_model_dir}")
        logger.error("Please ensure the CSM model files are present.")
        return

    # 2. Determine Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target device: {device}")
    if device == "cuda":
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    # 3. Load Generator
    generator = None
    sample_rate = 24000
    try:
        logger.info(f"Loading local CSM model from base path '{BASE_MODEL_PATH}'...")
        start_load = time.perf_counter()
        generator = load_csm_1b_local(model_path=BASE_MODEL_PATH, device=device)
        sample_rate = generator.sample_rate
        end_load = time.perf_counter()
        logger.info(f"Model and generator loaded successfully in {end_load - start_load:.2f} seconds.")
        logger.info(f"Using sample rate: {sample_rate}")
    except Exception as e:
        logger.error(f"Failed to load model or initialize generator: {e}", exc_info=True)
        return

    # --- Loop through Speaker IDs (Updated Range and Error Handling) ---
    success_count = 0
    fail_count = 0
    total_gen_time_ms = 0
    total_conv_time_ms = 0
    speakers_attempted = 0

    for speaker_id in range(MAX_SPEAKER_ID_TO_TEST + 1): # Loop from 0 to MAX_SPEAKER_ID_TO_TEST inclusive
        speakers_attempted += 1
        logger.info(f"\n--- Attempting Speaker ID: {speaker_id} ---")

        # 4. Generate Audio
        logger.info(f"Generating audio for: \"{TEST_TEXT}\"")
        audio_tensor = None
        generation_ms = -1.0
        output_filename_base = f"test_output_spk{speaker_id}"
        wav_filepath = OUTPUT_DIR / f"{output_filename_base}.wav"
        mp3_filepath = OUTPUT_DIR / f"{output_filename_base}.mp3"

        try:
            start_gen = time.perf_counter()
            audio_tensor = generator.generate(
                text=TEST_TEXT,
                speaker=speaker_id,
                context=[],
                max_audio_length_ms=15_000, # Keep max length relatively short for testing
                temperature=0.8,
                topk=50
            )
            end_gen = time.perf_counter()
            generation_ms = (end_gen - start_gen) * 1000
            total_gen_time_ms += generation_ms

            if audio_tensor is None or audio_tensor.numel() == 0:
                logger.error(f"Audio generation failed for speaker {speaker_id} (empty tensor returned). Stopping loop.")
                fail_count += 1
                break # Stop the loop if generation returns empty tensor

            logger.info(f"Audio generation successful in {generation_ms:.2f} ms.")

            # 5. Save Temp WAV and Convert to MP3
            conversion_ok, conversion_ms = save_temp_wav_and_convert_mp3(
                audio_tensor, sample_rate, mp3_filepath
            )

            if conversion_ok:
                logger.info(f"MP3 file created: {mp3_filepath.name}")
                if conversion_ms > 0: # Add conversion time if successful
                    total_conv_time_ms += conversion_ms
                success_count += 1
            else:
                logger.error(f"Failed to create MP3 for speaker {speaker_id}.")
                # Still count generation as success if WAV was okay, but MP3 failed?
                # Let's count success based on MP3 creation for simplicity here.
                fail_count += 1
                # Optionally stop if conversion fails? For now, let's continue
                # break

        except Exception as e:
            logger.error(f"--- Error during generation for Speaker ID: {speaker_id} ---", exc_info=True)
            logger.error(f"Stopping speaker iteration due to error.")
            fail_count += 1
            break # Stop the loop on any exception during generation

    # --- Final Summary ---
    logger.info("\n--- Multi-Speaker TTS Generation Test Finished ---")
    logger.info(f"Speakers Attempted: {speakers_attempted} (IDs 0 to {speakers_attempted - 1})")
    logger.info(f"Successful MP3 Generations: {success_count}")
    logger.info(f"Failed/Stopped Generations: {fail_count}")
    if fail_count > 0:
        logger.info(f"(Loop stopped early if Failures > 0)")
    avg_gen_time_s = (total_gen_time_ms / success_count) / 1000 if success_count > 0 else 0
    avg_conv_time_ms = (total_conv_time_ms / success_count) if success_count > 0 else 0
    logger.info(f"Average Generation Time (Successful): {avg_gen_time_s:.2f} seconds")
    logger.info(f"Average MP3 Conversion Time (Successful): {avg_conv_time_ms:.2f} ms")
    logger.info(f"Output files saved in: {OUTPUT_DIR}")
    logger.info("---")

if __name__ == "__main__":
    run_multi_tts_test()