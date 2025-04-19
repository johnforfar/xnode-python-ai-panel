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

# Force CPU usage for all operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor

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

class Generator:
    def __init__(self, model: Model, cache={}, device="cpu", max_batch_size=1):
        # Force CPU usage regardless of what's passed
        self.device = "cpu"
        
        self.model = model
        self.model.setup_caches(1)
        self._text_tokenizer = self.load_llama3_tokenizer()
        self.sample_rate = 24000
        
        # Load Mimi audio tokenizer from local files
        print(f"INFO:     Loading Mimi audio tokenizer from local files")
        
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Search for Mimi weight file in several locations
        potential_paths = [
            os.path.join(PROJECT_ROOT, "models/kyutai/mimi_weight.pt"),
            os.path.join(PROJECT_ROOT, "models/kyutai/mimi_weights.pt"),
            os.path.join(PROJECT_ROOT, "models/mimi_weight.pt"),
            os.path.join(PROJECT_ROOT, "models/mimi/mimi_weight.pt"),
            os.path.join(PROJECT_ROOT, "models/kyutai/mimi/mimi_weight.pt"),
        ]
        
        mimi_path = None
        for path in potential_paths:
            if os.path.exists(path):
                mimi_path = path
                print(f"INFO:     Found Mimi weights at {mimi_path}")
                break
        
        if mimi_path is None:
            paths_checked = "\n- ".join(potential_paths)
            raise FileNotFoundError(
                f"Mimi model file not found in any expected local locations:\n- {paths_checked}\n"
                "Please ensure 'mimi_weight.pt' is in one of these directories."
            )
        
        print(f"INFO:     Loading Mimi from {mimi_path}")
        
        # Load with diagnostic output
        print(f"INFO:     Loading Mimi weights...")
        
        # Force weights_only=False for PyTorch 2.6 compatibility
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Apply the patch
        torch.load = patched_torch_load
        
        try:
            from moshi.models import loaders
            mimi = loaders.get_mimi(mimi_path, device=self.device)
            mimi.set_num_codebooks(32)
            self._audio_tokenizer = mimi
            print(f"INFO:     Mimi audio tokenizer loaded successfully")
        except Exception as e:
            print(f"ERROR:    Failed to load Mimi: {e}")
            raise
        finally:
            # Restore original torch.load
            torch.load = original_torch_load

    def load_llama3_tokenizer(self):
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        llama_path = os.path.join(PROJECT_ROOT, "models/llama-3.2-1b")
        if not os.path.exists(llama_path):
            raise FileNotFoundError(
                f"Llama tokenizer not found at {llama_path}. "
                "Please download it to './models/llama-3.2-1b/'."
            )
        tokenizer = AutoTokenizer.from_pretrained(llama_path)
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )
        return tokenizer

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Tokenize audio into frames using Mimi"""
        audio_frames = self._audio_tokenizer.encode(audio.to(self.device))
        return audio_frames

    @torch.inference_mode()
    def generate(self, text: str, speaker: int, context: List[Segment], max_audio_length_ms: float = 10000, temperature: float = 0.9, topk: int = 50, output_mp3: bool = True, output_path: str = None):
        self.model.reset_caches()
        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_text_segment(segment.text, segment.speaker)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below {max_seq_len}")
        
        print(f"INFO:     Generating audio frames")
        
        for i in range(max_audio_frames):
            try:
                sample = self.model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    print(f"INFO:     Generation complete (found end token)")
                    break
                samples.append(sample)
                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
                
                if i % 10 == 0:
                    print(f"INFO:     Generated {i+1}/{max_audio_frames} frames")
                    
            except Exception as e:
                print(f"ERROR:    Frame generation error: {str(e)}")
                break
        
        print(f"INFO:     Generated {len(samples)} frames, decoding audio")
        
        # Decode the audio
        if samples:
            # Concatenate all frames
            samples_tensor = torch.cat(samples, dim=0)
            
            # Decode to audio
            audio = self._audio_tokenizer.decode(samples_tensor)
            
            # Normalize audio to avoid clipping
            if torch.abs(audio).max() > 0.99:
                print(f"INFO:     Normalizing audio to prevent clipping")
                max_amp = torch.abs(audio).max()
                if max_amp > 0:
                    audio = 0.9 * audio / max_amp
            
            print(f"INFO:     Audio successfully decoded, shape: {audio.shape}")
            
            # Convert to MP3 if requested
            if output_mp3 and output_path:
                # Determine output filenames
                base_filename, extension = os.path.splitext(output_path)
                wav_file = f"{base_filename}_temp.wav"
                mp3_file = f"{base_filename}.mp3"
                
                # Save temporary WAV file
                torchaudio.save(wav_file, audio.unsqueeze(0).cpu(), self.sample_rate)
                
                # Convert to MP3
                convert_to_mp3(wav_file, mp3_file)
                
                # Remove temporary WAV file
                try:
                    os.remove(wav_file)
                    print(f"INFO:     Removed temporary WAV file: {wav_file}")
                except Exception as e:
                    print(f"WARNING:  Could not remove temporary WAV file: {e}")
                
                # Return audio tensor
                return audio
            else:
                # Just return the audio tensor
                return audio

def load_csm_1b_local(device=None, model_dir=None):
    """
    Load the CSM-1B model from local files only (no downloads)
    
    Args:
        device: The device to load the model on (auto-detected if None)
        model_dir: The directory containing the model files
    
    Returns:
        Generator: A generator initialized with the CSM model
    """
    import os
    import torch
    import logging
    from pathlib import Path
    from models import Model, llama3_2_1B
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"INFO:     Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # M1 Mac has MPS but we'll force CPU for better compatibility
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            device = "cpu"
            print(f"INFO:     Detected M1 Mac, forcing CPU usage for better compatibility")
        else:
            device = "cpu"
            print(f"INFO:     Using CPU")
    
    # For M1 Macs, force CPU usage regardless
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = "cpu"
        print(f"INFO:     Overriding to CPU for M1 Mac")
    
    logger = logging.getLogger(__name__)
    
    # Setup model directory paths
    if model_dir is None:
        # Try multiple possible model locations
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        potential_model_dirs = [
            os.path.join(PROJECT_ROOT, "models"),
            Path("/models"),
            Path("models"),
            Path("../models"),
            Path("../../models"),
        ]
        
        for path in potential_model_dirs:
            path = Path(path)
            if path.exists():
                model_dir = path
                logger.info(f"Found models directory at {path}")
                break
                
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, "models")
            logger.warning(f"No models directory found, defaulting to {model_dir}")
    
    model_dir = Path(model_dir)
    logger.info(f"Loading CSM model from local directory: {model_dir}")
    
    # Check if model directory exists
    if not model_dir.exists():
        logger.warning(f"Model directory {model_dir} does not exist, will try to continue anyway")
    
    # Search for CSM model file in multiple potential locations
    csm_patterns = [
        "csm_1b.pt", 
        "csm-1b.pt", 
        "csm_weights.pt", 
        "csm/csm_weights.pt",
        "csm-1b/csm_1b.pt",
        "csm-1b/csm-1b.pt",
        "csm-1b/csm_weights.pt"
    ]
    
    # Try both the specified model_dir and PROJECT_ROOT/models
    all_paths_to_try = [model_dir]
    if model_dir != Path(os.path.join(PROJECT_ROOT, "models")):
        all_paths_to_try.append(Path(os.path.join(PROJECT_ROOT, "models")))
    
    csm_model_path = None
    # Try all combinations of paths and patterns
    for base_path in all_paths_to_try:
        for pattern in csm_patterns:
            path = base_path / pattern
            if path.exists():
                csm_model_path = path
                logger.info(f"Found CSM model at {path}")
                break
        if csm_model_path is not None:
            break
    
    if csm_model_path is None:
        raise FileNotFoundError(f"CSM model file not found in {model_dir}")
    
    # Initialize the model architecture (without weights)
    logger.info("Initializing CSM model architecture")
    model = Model(llama3_2_1B())
    model = model.to(device)
    
    # Initialize the generator
    generator = Generator(model=model, device=device)
    
    # Load model weights
    logger.info(f"Loading CSM weights from {csm_model_path}")
    try:
        # Load weights with weights_only=False to handle older versions of PyTorch
        state_dict = torch.load(csm_model_path, map_location=device, weights_only=False)
        generator.model.load_state_dict(state_dict)
        logger.info("CSM weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CSM weights: {e}")
        try:
            # Try with weights_only omitted
            state_dict = torch.load(csm_model_path, map_location=device)
            generator.model.load_state_dict(state_dict)
            logger.info("CSM weights loaded successfully with alternative method")
        except Exception as e2:
            logger.error(f"All attempts to load CSM weights failed: {e2}")
            raise
    
    logger.info("Local CSM generator initialized successfully")
    return generator

# Alias for backward compatibility 
load_csm_1b = load_csm_1b_local