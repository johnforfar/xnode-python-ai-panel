# ./python-app/src/generator.py
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import load_watermarker, watermark, CSM_1B_GH_WATERMARK
import os

@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor

class Generator:
    def __init__(self, model: Model):
        self._model = model
        self._model.setup_caches(1)
        self._text_tokenizer = self.load_llama3_tokenizer()
        self.device = next(model.parameters()).device
        self.sample_rate = 24000
        
        # Load Mimi audio tokenizer strictly from local files
        print(f"INFO:     Loading Mimi audio tokenizer from local files")
        try:
            from rustymimi import MimiCodec as RustMimi
            print(f"INFO:     Using RustMimi audio tokenizer")
            mimi = RustMimi()
            mimi.set_num_codebooks(32)
            self._audio_tokenizer = mimi
            print(f"INFO:     RustMimi audio tokenizer loaded successfully")
        except ImportError:
            print(f"INFO:     RustMimi not available, falling back to PyTorch Mimi")
            mimi_paths = [
                os.path.join("./models/kyutai", "mimi_weight.pt"),  # Your confirmed working file
                os.path.join("./models/kyutai/mimi", "mimi_weight.pt"),
                os.path.join("./models/kyutai", "model.safetensors"),  # Fallback to your original file
                os.path.join("./models", "mimi_weight.pt"),
                os.path.join("./models/csm-1b", "mimi_weight.pt")
            ]
            
            mimi_path = None
            for path in mimi_paths:
                if os.path.exists(path):
                    mimi_path = path
                    print(f"INFO:     Found Mimi at path: {mimi_path}")
                    break
            
            if mimi_path is None:
                raise FileNotFoundError(
                    "Mimi model file not found in any expected local locations. "
                    "Ensure 'mimi_weight.pt' or 'model.safetensors' is in './models/kyutai/'."
                )
            
            print(f"INFO:     Loading Mimi from {mimi_path}")
            mimi = loaders.get_mimi(mimi_path, device=self.device)
            mimi.set_num_codebooks(32)
            self._audio_tokenizer = mimi
            print(f"INFO:     Mimi audio tokenizer loaded successfully")
        
        # Load watermarker
        self._watermarker = load_watermarker(device=self.device)

    def load_llama3_tokenizer(self):
        llama_path = os.path.join("./models", "llama-3.2-1b")
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
    def generate(self, text: str, speaker: int, context: List[Segment], max_audio_length_ms: float = 10000, temperature: float = 0.9, topk: int = 50):
        self._model.reset_caches()
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
        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break
            samples.append(sample)
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        
        if not samples:
            print(f"WARNING:  No audio frames generated, returning silent audio")
            audio_length = int(self.sample_rate * (max_audio_length_ms / 1000))
            return torch.zeros(audio_length, dtype=torch.float32)
        
        samples_tensor = torch.cat(samples, dim=0)
        
        print(f"INFO:     Decoding audio from {len(samples)} frames using Mimi")
        try:
            audio = self._audio_tokenizer.decode(samples_tensor)
            print(f"INFO:     Applying watermark to audio")
            watermarked_audio = watermark(audio, self._watermarker, CSM_1B_GH_WATERMARK)
            return watermarked_audio
        except Exception as e:
            print(f"ERROR:    Failed to decode audio: {str(e)}")
            audio_length = int(self.sample_rate * (max_audio_length_ms / 1000))
            return torch.zeros(audio_length, dtype=torch.float32)

def load_csm_1b_local(device: str = "cpu", model_dir: str = "./models") -> Generator:
    """Load CSM 1B model from local directory with detailed logging"""
    print(f"DEBUG: Starting CSM model loading from {model_dir}")
    
    # Set environment variables for offline mode
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = model_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = model_dir
    os.environ["NO_TORCH_COMPILE"] = "1"
    
    csm_path = os.path.join(model_dir, "csm-1b")
    llama_path = os.path.join(model_dir, "llama-3.2-1b")
    
    print(f"DEBUG: Checking for CSM model: {csm_path} - {'EXISTS' if os.path.exists(csm_path) else 'NOT FOUND'}")
    print(f"DEBUG: Checking for Llama model: {llama_path} - {'EXISTS' if os.path.exists(llama_path) else 'NOT FOUND'}")
    
    if not os.path.exists(csm_path):
        raise ValueError(f"CSM model not found at {csm_path}")
    if not os.path.exists(llama_path):
        raise ValueError(f"Llama model not found at {llama_path}")
    
    try:
        print(f"DEBUG: Creating model config")
        config = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=2051,
            audio_num_codebooks=32
        )
        
        print(f"DEBUG: Initializing model with config")
        model = Model(config)
        
        model_path = os.path.join(csm_path, "ckpt.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"CSM model checkpoint not found at {model_path}")
        
        print(f"DEBUG: Loading model from checkpoint: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=torch.float32)
        print(f"DEBUG: Model moved to device {device}")
        
        return Generator(model)
    except Exception as e:
        print(f"DEBUG: Exception during model loading: {str(e)}")
        raise