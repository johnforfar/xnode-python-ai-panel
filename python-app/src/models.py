# ./python-app/src/models.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import os
from pathlib import Path
import logging
from safetensors.torch import load_file as load_safetensors_file
import pickle

# Ensure torchtune is installed: pip install torchtune
try:
    import torchtune
    from torchtune.models import llama3_2
except ImportError:
    print("ERROR: torchtune library not found. Please install it: pip install torchtune")
    raise

logger = logging.getLogger(__name__)

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
def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    # Define parameters for the 1B model based on the original
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    # Define parameters for the 100M model based on the original
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,  # Adjust if decoder max_seq_len is different
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,  # Adjust if needed for decoder
        scale_factor=32,  # Adjust if needed for decoder
    )

FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}

def _prepare_transformer(model):
    # Check if tok_embeddings exists before accessing embedding_dim
    embed_dim = getattr(model, 'tok_embeddings', None)
    if embed_dim is not None:
        embed_dim = model.tok_embeddings.embedding_dim
        model.tok_embeddings = nn.Identity()
    else:  # Fallback or error if structure is unexpected
        logger.warning("Model structure might differ from expected (torchtune). Assuming embed_dim from config.")
        embed_dim = model.embed_dim  # Assuming it's directly accessible
    model.output = nn.Identity()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, embed_dim

def _create_causal_mask(seq_len: int, device: torch.device):
    # Creates a lower triangular mask for causal attention
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Indexes a square causal mask using a tensor of indices.
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)
    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    try:
        r = mask[input_pos, :]
        return r
    except IndexError as e:
        raise

def _multinomial_sample_one_no_sync(probs):
    # Efficient multinomial sampling
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    # Samples from the top-k logits
    logits = logits / temperature
    filter_value: float = -float("Inf")
    k = min(topk, logits.size(-1))
    if k <= 0:
        k = 1  # Sample at least the top token
    top_k_logits, _ = torch.topk(logits, k)
    min_top_k_logit = top_k_logits[..., -1, None]  # Get the k-th logit value

    indices_to_remove = logits < min_top_k_logit
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    if torch.all(scores_processed == filter_value):
        logger.warning("All logits filtered out in sample_topk. Returning argmax of original logits.")
        return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.int)

    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token

@dataclass
class ModelArgs:
    backbone_flavor: str = "llama-1B"
    decoder_flavor: str = "llama-100M"
    text_vocab_size: int = 128256
    audio_vocab_size: int = 2051  # Corrected based on state_dict mismatch errors
    audio_num_codebooks: int = 32

# Handle PyTorchModelHubMixin availability
try:
    from huggingface_hub import PyTorchModelHubMixin
    BaseModelClass = (nn.Module, PyTorchModelHubMixin)
except ImportError:
    logger.warning("huggingface_hub not found. PyTorchModelHubMixin features unavailable.")
    BaseModelClass = (nn.Module,)

class Model(*BaseModelClass):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        logger.info(f"Initializing Model with config: {config}")
        logger.info(f"Initializing backbone: {config.backbone_flavor}")
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())

        logger.info(f"Initializing decoder: {config.decoder_flavor}")
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        logger.info("Initializing embeddings and heads...")
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        audio_total_embeddings = config.audio_vocab_size * config.audio_num_codebooks
        self.audio_embeddings = nn.Embedding(audio_total_embeddings, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))
        logger.info("Model components initialized (KV Caching ENABLED).")

    def setup_caches(self, max_batch_size: int) -> None:
        """Setup KV caches with device context"""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        logger.info(f"Setting up caches for batch_size={max_batch_size}, device={device}, dtype={dtype}")

        with device:
            self.backbone.setup_caches(max_batch_size)
            self.decoder.setup_caches(max_batch_size, decoder_max_seq_len=self.config.audio_num_codebooks)

        backbone_max_seq_len = self.backbone.max_seq_len
        decoder_mask_len = self.config.audio_num_codebooks
        self.register_buffer("backbone_causal_mask", _create_causal_mask(backbone_max_seq_len, device), persistent=False)
        self.register_buffer("decoder_causal_mask", _create_causal_mask(decoder_mask_len, device), persistent=False)
        logger.debug("Causal masks registered successfully.")

    def reset_caches(self):
        if hasattr(self.backbone, 'reset_caches'):
            self.backbone.reset_caches()
        if hasattr(self.decoder, 'reset_caches'):
            self.decoder.reset_caches()

    def generate_frame(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor, temperature: float, topk: int) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # Backbone processing
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        curr_seq_len = input_pos.size(1)
        indexed_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        curr_backbone_mask = indexed_mask[:, :, :curr_seq_len]
        backbone_output = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        h = backbone_output.to(dtype=dtype)

        # Decoder processing
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos_decoder = torch.arange(0, curr_h.size(1), device=device).unsqueeze(0).repeat(curr_h.size(0), 1)

        for i in range(1, self.config.audio_num_codebooks):
            if hasattr(self.decoder, 'reset_caches'):
                self.decoder.reset_caches()
            projected_input = self.projection(curr_h)
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos_decoder)
            decoder_output = self.decoder(projected_input, input_pos=curr_pos_decoder, mask=curr_decoder_mask)
            decoder_h = decoder_output.to(dtype=dtype)

            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos_decoder = curr_pos_decoder[:, -1:] + 1

        return curr_sample

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        offset = codebook * self.config.audio_vocab_size
        return self.audio_embeddings(tokens + offset)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        audio_tokens = tokens[:, :, :-1]
        codebook_indices = torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        audio_tokens_with_offset = audio_tokens + (self.config.audio_vocab_size * codebook_indices)
        b, s, cb = audio_tokens_with_offset.shape
        audio_embeds = self.audio_embeddings(audio_tokens_with_offset.view(b * s, cb))
        audio_embeds = audio_embeds.view(b, s, cb, -1)
        return torch.cat([audio_embeds, text_embeds], dim=2)

    @classmethod
    def from_local_pretrained(cls, local_path: str):
        local_path = Path(local_path)
        logger.info(f"Loading model from local path: {local_path}")
        config = ModelArgs()
        model = cls(config)

        potential_weight_files = ["model.safetensors", "pytorch_model.bin", "csm_weights.pt", "csm-1b.pt"]
        for filename in potential_weight_files:
            path_to_check = local_path / filename
            if path_to_check.is_file():
                weight_file_path = path_to_check
                weight_file_type = "safetensors" if filename.endswith(".safetensors") else "pytorch"
                logger.info(f"Found weight file: {weight_file_path} (Type: {weight_file_type})")
                break
        else:
            raise FileNotFoundError(f"Could not find model weight file in {local_path}")

        map_location_device = 'cpu'
        if weight_file_type == "safetensors":
            state_dict = load_safetensors_file(weight_file_path, device=map_location_device)
        else:
            state_dict = torch.load(weight_file_path, map_location=map_location_device, weights_only=True)

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        logger.info(f"Model loaded to CPU and set to eval mode. Device transfer happens after.")
        return model