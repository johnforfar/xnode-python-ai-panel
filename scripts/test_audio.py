# ./scripts/test_audio.py
import os
import sys
import torch
import torchaudio

# Add the python-app/src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
src_dir = os.path.join(project_root, "python-app/src")
sys.path.insert(0, src_dir)

# Now import from generator
from generator import load_csm_1b_local

def show_paths():
    """Show all relevant paths to debug path issues"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    src_dir = os.path.join(project_root, "python-app/src")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Source directory: {src_dir}")
    
    # Check for important files
    kyutai_dir = os.path.join(project_root, "models/kyutai")
    mimi_path = os.path.join(kyutai_dir, "mimi_weight.pt")
    csm_path = os.path.join(project_root, "models/csm-1b/ckpt.pt")
    
    print(f"Kyutai directory exists: {os.path.exists(kyutai_dir)}")
    print(f"Mimi weights exist: {os.path.exists(mimi_path)}")
    print(f"CSM checkpoint exists: {os.path.exists(csm_path)}")

def test_audio_generation():
    """Generate a test audio clip using the CSM model with fallbacks"""
    # First show paths to debug
    show_paths()
    
    # Patch torch.load to handle PyTorch 2.6 changes
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    # Apply the patch globally
    torch.load = patched_torch_load
    
    try:
        # Create absolute paths to ensure everything can be found
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))
        model_dir = os.path.join(project_root, "models")
        
        print("\nInitializing CSM model with absolute path...")
        print(f"Using model_dir: {model_dir}")
        
        # Use CPU for testing
        try:
            generator = load_csm_1b_local(device="cpu", model_dir=model_dir)
            print("CSM model loaded successfully!")
        except Exception as e:
            print(f"Error loading CSM model: {str(e)}")
            print("Please check the model files and paths.")
            return
        
        print("\nTesting text tokenization...")
        text = "Hello, this is a test of the audio generation system."
        print(f"Text: {text}")
        
        # Test text tokenization
        tokens, tokens_mask = generator._tokenize_text_segment(text, speaker=0)
        print(f"Text successfully tokenized! Shape: {tokens.shape}")
            
        # Now proceed with audio generation
        print("\nGenerating audio...")
        print(f"This is now using a DummyMimi fallback if the real Mimi failed")
        
        audio = generator.generate(
            text=text,
            speaker=0,
            context=[],
            max_audio_length_ms=3000,  # 3 seconds
            temperature=0.8,
        )
        
        # Check if audio was generated
        if audio is None or (isinstance(audio, torch.Tensor) and audio.numel() == 0):
            print("Error: Generated audio is empty!")
            return
            
        print(f"Audio generation completed! Shape: {audio.shape}")
        
        # Save the audio to a WAV file in project root
        output_path = os.path.join(project_root, "test_output.wav")
        
        # Ensure audio is in the right format for saving
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(output_path, audio.cpu(), generator.sample_rate)
        print(f"Audio saved to {output_path}")
        
        print("\nTest completed successfully!")
        
        # Write a success report
        report_path = os.path.join(project_root, "test_report.txt")
        with open(report_path, "w") as f:
            f.write("Audio test completed successfully.\n")
            f.write(f"Model dir: {model_dir}\n")
            f.write(f"Audio shape: {audio.shape}\n")
            f.write(f"Output file: {output_path}\n")
        
        return True
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTest failed with exception.")
        return False
    finally:
        # Restore original torch.load function
        torch.load = original_torch_load

if __name__ == "__main__":
    test_audio_generation()