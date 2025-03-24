# ./python-app/src/sesame_tts.py
import os
import torch
import torchaudio
import asyncio
import tempfile
import subprocess
from datetime import datetime
from generator import load_csm_1b, Segment, load_csm_1b_local

class SesameTTS:
    def __init__(self, device="cpu", model_dir="/models"):
        """Initialize the Sesame CSM-1B TTS system"""
        print(f"INFO:     Initializing SesameTTS with device={device}, model_dir={model_dir}")
        
        # Disable Triton compilation which can cause issues
        os.environ["NO_TORCH_COMPILE"] = "1"
        
        try:
            # Set environment variables for HuggingFace to find models locally
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HOME"] = model_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = model_dir
            
            # Set the paths to the model explicitly
            print(f"INFO:     Loading CSM model from {model_dir}")
            self.generator = load_csm_1b_local(device=device, model_dir=model_dir)
            self.sample_rate = self.generator.sample_rate
            print(f"INFO:     CSM model loaded successfully, sample_rate={self.sample_rate}")
        except Exception as e:
            print(f"ERROR:    Failed to initialize CSM model: {str(e)}")
            # Create a fallback option so the app doesn't crash
            self.generator = None
            self.sample_rate = 24000  # Default sample rate for CSM
            print(f"INFO:     Using fallback audio generator")
    
    async def generate_audio(self, text, speaker_id=0):
        """Generate audio from text using CSM-1B or fallback to silent audio if needed"""
        print(f"INFO:     Generating audio with text: '{text[:50]}...' speaker_id={speaker_id}")
        
        if self.generator is None:
            # If CSM model failed to load, generate silent audio
            print(f"INFO:     Using silent audio fallback (CSM model not available)")
            silent_audio = torch.zeros(self.sample_rate)
            return silent_audio
        
        try:
            # Try to generate real audio with CSM
            print(f"INFO:     Attempting to generate audio using CSM model")
            audio = self.generator.generate(
                text=text,
                speaker=speaker_id,
                context=[],  # No context for now
                max_audio_length_ms=10000,  # 10 seconds max
                temperature=0.8,
            )
            
            print(f"INFO:     Audio generation successful, shape={audio.shape}")
            return audio
            
        except Exception as e:
            print(f"ERROR:    CSM audio generation failed: {str(e)}")
            print(f"INFO:     Falling back to silent audio")
            
            # Fallback to silent audio
            silent_audio = torch.zeros(self.sample_rate)
            return silent_audio

# Example usage
if __name__ == "__main__":
    import asyncio
    # Force CPU usage explicitly
    tts = SesameTTS(device="cpu", model_dir="/models")
    audio = asyncio.run(tts.generate_audio("Hello, how are you?", speaker_id=0))
    torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), tts.generator.sample_rate)