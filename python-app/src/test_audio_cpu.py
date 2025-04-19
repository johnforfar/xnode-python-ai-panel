import os
# Force CPU usage for all operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from generator import load_csm_1b
import torchaudio
import torch
import os.path
from subprocess import run

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

# Force CPU usage
device = "cpu"
print(f"Using device: {device}")

# Load the generator
print("Loading CSM model on CPU. This may take a while...")
generator = load_csm_1b(device=device)

# Text to synthesize
text = "Welcome to the OpenxAI Panel!"

# Try the first 10 speaker IDs
num_speakers = 10
for speaker_id in range(num_speakers):
    try:
        print(f"\nGenerating audio for speaker {speaker_id}...")
        
        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=20_000,
        )
        
        # Create a unique filename if the file already exists
        base_filename = f"audio-speaker{speaker_id}"
        wav_extension = ".wav"
        mp3_extension = ".mp3"
        
        # WAV filename
        wav_file = f"{base_filename}{wav_extension}"
        counter = 1
        while os.path.exists(wav_file):
            wav_file = f"{base_filename}-{counter:03d}{wav_extension}"
            counter += 1
            
        # MP3 filename (based on WAV filename)
        mp3_file = wav_file.replace(wav_extension, mp3_extension)
        
        # Save WAV file
        torchaudio.save(wav_file, audio.unsqueeze(0).cpu(), generator.sample_rate)
        print(f"Audio generated and saved to {wav_file}")
        
        # Convert to MP3
        convert_to_mp3(wav_file, mp3_file)
        
    except Exception as e:
        print(f"Error generating audio for speaker {speaker_id}: {e}")
        print(f"Skipping to next speaker...")
        continue

print("\nAll speaker generations completed!") 
