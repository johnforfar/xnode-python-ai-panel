# ./python-app/src/models.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import os
from pathlib import Path

# Define PROJECT_ROOT at the module level
PROJECT_ROOT = Path(__file__).parent.parent.parent
print(f"INFO: [models.py] PROJECT_ROOT set to: {PROJECT_ROOT}")

# Custom Embedding class to skip random initialization
class UninitializedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, input):
        return nn.functional.embedding(input, self.weight)

# Define model configurations
def llama_3_2_1b():  # Fixed naming to match what's imported
    return {
        "vocab_size": 128256,
        "num_layers": 16,
        "num_heads": 32,
        "num_kv_heads": 8,
        "embed_dim": 2048,
        "max_seq_len": 2048,
        "intermediate_dim": 8192,
        "attn_dropout": 0.0,
        "norm_eps": 1e-5,
        "rope_base": 500000,
        "scale_factor": 32
    }

def llama_3_2_100m():
    return {
        "vocab_size": 128256,
        "num_layers": 4,
        "num_heads": 8,
        "num_kv_heads": 2,
        "embed_dim": 1024,
        "max_seq_len": 2048,
        "intermediate_dim": 8192,
        "attn_dropout": 0.0,
        "norm_eps": 1e-5,
        "rope_base": 500000,
        "scale_factor": 32
    }

FLAVORS = {
    "llama-1B": llama_3_2_1b,
    "llama-100M": llama_3_2_100m,
}

def _prepare_transformer(model_config):
    embed_dim = model_config["embed_dim"]
    return model_config, embed_dim

def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    return mask[input_pos, :]

def _multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, -float("Inf"))
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    return _multinomial_sample_one_no_sync(probs)

@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int

    @classmethod
    def from_local_config(cls, model_dir: str = None):
        """Load model configuration from local directory"""
        if model_dir is None:
            # Try to find models in the repository structure
            PROJECT_ROOT = Path(__file__).parent.parent.parent
            model_dir = PROJECT_ROOT / "models"
        
        # Default configuration for CSM-1B
        return cls(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,  # Standard for Llama tokenizer
            audio_vocab_size=1024,   # Standard for CSM-1B
            audio_num_codebooks=32   # Standard for CSM-1B
        )

def load_local_model(model_path: str = None):
    """Load a model from local files only"""
    if model_path is None:
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        potential_paths = [
            PROJECT_ROOT / "models" / "csm-1b" / "csm_weights.pt",
            PROJECT_ROOT / "models" / "csm-1b" / "csm-1b.pt",
            PROJECT_ROOT / "models" / "csm-1b.pt",
            Path("/models/csm-1b/csm_weights.pt"),
        ]
        
        for path in potential_paths:
            if path.exists():
                model_path = path
                print(f"Found model at: {model_path}")
                break
    
    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(f"Could not find model file in expected locations")
    
    # Initialize model with default args
    args = ModelArgs.from_local_config()
    model = Model(args)
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model

# Export the functions that are imported elsewhere
llama3_2_1B = llama_3_2_1b  # Alias for backward compatibility

class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())
        self.text_embeddings = UninitializedEmbedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = UninitializedEmbedding(config.audio_vocab_size * config.audio_num_codebooks, backbone_dim)
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))

    def setup_caches(self, max_batch_size: int):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype)
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device))

    def generate_frame(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor, temperature: float, topk: int):
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
        self.decoder.reset_caches()
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1
        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor):
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor):
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)