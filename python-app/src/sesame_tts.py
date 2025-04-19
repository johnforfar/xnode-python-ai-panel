# ./python-app/src/sesame_tts.py
import os
import torch
import torchaudio
import asyncio
import tempfile
import subprocess
from datetime import datetime
from generator import load_csm_1b_local

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
        """Initialize the Sesame CSM-1B TTS system"""
        print(f"INFO:     Initializing SesameTTS with device={device}, model_dir={model_dir}")
        
        # Disable Triton compilation which can cause issues
        os.environ["NO_TORCH_COMPILE"] = "1"
        
        # Determine model directory relative to project root if not absolute
        if model_dir is None or not os.path.isabs(model_dir):
            from models import PROJECT_ROOT # Import from models.py
            resolved_model_dir = PROJECT_ROOT / (model_dir or "models")
            print(f"INFO:     Resolved model directory to: {resolved_model_dir}")
        else:
            resolved_model_dir = model_dir
        
        try:
            # Set environment variables for HuggingFace to find models locally
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HOME"] = str(resolved_model_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(resolved_model_dir)
            
            # Set the paths to the model explicitly
            print(f"INFO:     Loading CSM model from {resolved_model_dir}")
            self.generator = load_csm_1b_local(device=device, model_dir=resolved_model_dir)
            self.sample_rate = self.generator.sample_rate
            print(f"INFO:     CSM model loaded successfully, sample_rate={self.sample_rate}")
            self.tts_available = True # Assume success if loader doesn't raise exception
        except Exception as e:
            print(f"ERROR:    Failed to initialize CSM model: {str(e)}")
            self.generator = None
            self.sample_rate = 24000  # Default sample rate for CSM
            self.tts_available = False
            print(f"INFO:     Using fallback audio generator")
    
    async def generate_audio_and_convert(self, text, speaker_id=0, output_dir="static/audio"):
        """Generates audio, saves as WAV, converts to MP3, returns MP3 path."""
        if not self.tts_available or self.generator is None:
            print("WARN: TTS not available, skipping audio generation.")
            return None # Indicate failure

        try:
            start_time = asyncio.get_event_loop().time()
            print(f"INFO: Generating audio for speaker {speaker_id}: '{text[:30]}...'")

            # --- Call the actual generate method of the loaded generator ---
            # This depends on the object returned by load_csm_1b_local
            # Assuming it has a .generate() method matching the script's usage
            audio_tensor = self.generator.generate(
                text=text,
                speaker=speaker_id,
                context=[], # Provide context if needed
                max_audio_length_ms=20_000, # Or adjust as needed
            )
            # --- End generate call ---

            generation_time = (asyncio.get_event_loop().time() - start_time) * 1000
            print(f"INFO: Audio tensor generated in {generation_time:.2f} ms, shape={audio_tensor.shape}")

            # Create unique filename
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            base_filename = f"audio_spk{speaker_id}_{timestamp}"
            wav_file = os.path.join(output_dir, f"{base_filename}.wav")
            mp3_file = os.path.join(output_dir, f"{base_filename}.mp3")

            # Save WAV
            # Ensure tensor is on CPU and has correct shape [channels, samples]
            # CSM output might be [1, samples] or [samples], adjust accordingly
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0) # Add channel dim if needed
            torchaudio.save(wav_file, audio_tensor.cpu(), self.sample_rate)
            print(f"Audio saved to {wav_file}")

            # Convert to MP3
            if convert_to_mp3(wav_file, mp3_file):
                # Optionally remove WAV file after successful conversion
                try:
                    os.remove(wav_file)
                    print(f"Removed temporary WAV: {wav_file}")
                except OSError as e:
                    print(f"Warning: Could not remove temp WAV file: {e}")
                return mp3_file # Return path to the MP3
            else:
                print("ERROR: MP3 conversion failed.")
                # Decide fallback: return WAV path or None?
                return None # Indicate failure if MP3 is required

        except AttributeError as e:
             print(f"ERROR: Generator object missing expected method (e.g., 'generate'): {e}")
             return None
        except Exception as e:
            print(f"ERROR: Audio generation/conversion failed: {str(e)}")
            return None # Indicate failure

# Example usage
if __name__ == "__main__":
    import asyncio
    # Force CPU usage explicitly
    tts = SesameTTS(device="cpu", model_dir="/models")
    audio = asyncio.run(tts.generate_audio_and_convert("Hello, how are you?", speaker_id=0))
    torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), tts.generator.sample_rate)