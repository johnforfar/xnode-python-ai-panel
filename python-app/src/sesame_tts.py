# ./python-app/src/sesame_tts.py
import os
import torch
import torchaudio
import asyncio
import tempfile
import subprocess
from datetime import datetime

class SesameTTS:
    def __init__(self, device="cpu", model_dir=None):
        """Initialize the Sesame CSM-1B TTS system"""
        print(f"INFO:     Initializing SesameTTS with device={device}")
        
        # Disable Triton compilation which can cause issues
        os.environ["NO_TORCH_COMPILE"] = "1"
        
        # For now, we'll create a dummy generator and sample rate
        # This avoids the tensor size mismatch error
        self.generator = None
        self.sample_rate = 24000  # Default sample rate for CSM
        print(f"INFO:     SesameTTS initialized successfully, sample_rate={self.sample_rate}")
        
    async def generate_audio(self, text, speaker_id=0):
        """
        Generate audio from text using fallback methods since CSM-1B 
        has compatibility issues
        """
        print(f"INFO:     Generating audio with text: '{text[:50]}...' speaker_id={speaker_id}")
        
        # Create a dummy audio tensor
        # This is a silent audio of 1 second
        dummy_audio = torch.zeros(self.sample_rate)
        
        # Add some noise to make it not completely silent
        # This is just a placeholder until we can fix the actual CSM integration
        dummy_audio += torch.randn_like(dummy_audio) * 0.01
        
        print(f"INFO:     Created placeholder audio, shape={dummy_audio.shape}")
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        return dummy_audio

# Example usage
if __name__ == "__main__":
    import asyncio
    # Force CPU usage explicitly
    tts = SesameTTS(device="cpu", model_dir="/Users/johnny/Code/OpenxAI/models")
    audio = asyncio.run(tts.generate_audio("Hello, how are you?", speaker_id=0))
    torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), tts.generator.sample_rate)