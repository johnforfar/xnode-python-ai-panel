# ./python-app/src/generator.py
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
# from moshi.models import loaders  # Commented out (GPU-specific)
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
# from watermarking import load_watermarker, watermark, CSM_1B_GH_WATERMARK  # Commented out (GPU-specific)

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
        self.sample_rate = 24000  # Default sample rate for placeholder audio
        # mimi_weight = hf_hub_download("sesame/csm-1b", "mimi_weight.pt")  # Commented out (GPU-specific)
        # mimi = loaders.get_mimi(mimi_weight, device=device)  # Commented out (GPU-specific)
        # mimi.set_num_codebooks(32)  # Commented out (GPU-specific)
        # self._audio_tokenizer = mimi  # Commented out (GPU-specific)
        # self._watermarker = load_watermarker(device=device)  # Commented out (GPU-specific)

    def load_llama3_tokenizer(self):
        tokenizer_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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
        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break
            samples.append(sample)
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        audio_length = int(self.sample_rate * (max_audio_length_ms / 1000))
        audio = torch.zeros(audio_length, dtype=torch.float32)
        return audio

def load_csm_1b(device: str = "cpu") -> "Generator":
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,  # Updated to match checkpoint
        audio_num_codebooks=32
    )
    model = Model(config)
    model_path = hf_hub_download("sesame/csm-1b", "ckpt.pt")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)  # Added weights_only=True
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=torch.float32)
    return Generator(model)