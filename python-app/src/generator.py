# ./python-app/src/generator.py
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
from models import Model, ModelArgs, llama3_2_1B
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, ClapModel
import os
import json
from pathlib import Path
import logging
from subprocess import run
from huggingface_hub import hf_hub_download
from moshi.models import loaders as moshi_loaders

# Force CPU usage for all operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Use PROJECT_ROOT defined in models.py or redefine if needed
try:
    from models import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
print(f"INFO: [generator.py] PROJECT_ROOT set to: {PROJECT_ROOT}")

logger = logging.getLogger(__name__)

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
            mimi = moshi_loaders.get_mimi(mimi_path, device=self.device)
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

def load_csm_1b(device: str = "cpu", model_dir: Path = None) -> Generator:
    """Loads the CSM model locally using Model.from_local_pretrained and returns the Generator."""
    if model_dir is None:
        model_dir = PROJECT_ROOT / "models"
    logger.info(f"Attempting to load CSM-1B model from local directory: {model_dir}")

    csm_model_path = model_dir / "csm-1b" # Path to the directory

    if not csm_model_path.exists():
        logger.error(f"CSM model directory not found: {csm_model_path}")
        raise FileNotFoundError(f"CSM model directory not found at {csm_model_path}")

    # --- Load using Model.from_local_pretrained ---
    try:
        logger.info(f"Calling Model.from_local_pretrained for path: {csm_model_path}")
        # This uses the method defined in models.py which handles safetensors/pytorch loading
        model = Model.from_local_pretrained(str(csm_model_path), device=device)

        # Optional: Convert to bfloat16 if needed
        if device != 'cpu' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
             logger.info("Converting loaded model to bfloat16.")
             model = model.to(torch.bfloat16)

    except Exception as e:
        # Log the specific error from our loading function
        logger.error(f"Failed to load model using Model.from_local_pretrained from {csm_model_path}: {e}", exc_info=True)
        raise # Re-raise the underlying loading error

    # Initialize the Generator class with the loaded model
    try:
        logger.info("Initializing Generator class with loaded model...")
        # Pass the model object (already loaded)
        generator = Generator(model, device=device)
        logger.info("Generator initialized successfully.")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize Generator class after loading model: {e}", exc_info=True)
        raise