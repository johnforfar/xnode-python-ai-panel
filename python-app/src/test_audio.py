# ./python-app/src/test_audio.py
import torch
import torchaudio
import os
from generator import load_csm_1b_local

def test_audio_generation():
    print("Initializing CSM model...")
    # Use CPU for testing
    generator = load_csm_1b_local(device="cpu", model_dir="./models")
    
    print("Generating audio...")
    audio = generator.generate(
        text="Hello, this is a test of the CSM audio generation system. How are you today?",
        speaker=0,
        context=[],
        max_audio_length_ms=5000,  # 5 seconds
        temperature=0.8,
    )
    
    print(f"Audio generation successful! Shape: {audio.shape}")
    
    # Save the audio to a WAV file
    output_path = "test_output.wav"
    torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {output_path}")

if __name__ == "__main__":
    test_audio_generation()