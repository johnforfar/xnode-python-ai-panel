# ./python-app/src/models.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import os
from pathlib import Path
import logging
from safetensors.torch import load_file as load_safetensors_file

# Ensure torchtune is installed: pip install torchtune
try:
    import torchtune
    from torchtune.models import llama3_2
except ImportError:
    print("ERROR: torchtune library not found. Please install it: pip install torchtune")
    raise

# Define PROJECT_ROOT for potential use in loading
PROJECT_ROOT = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)
logger.info(f"INFO: [models.py] PROJECT_ROOT set to: {PROJECT_ROOT}")

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
        max_seq_len=2048, # Adjust if decoder max_seq_len is different
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000, # Adjust if needed for decoder
        scale_factor=32, # Adjust if needed for decoder
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
    else: # Fallback or error if structure is unexpected
        logger.warning("Model structure might differ from expected (torchtune). Assuming embed_dim from config.")
        # We might need to get embed_dim differently if tok_embeddings isn't there
        embed_dim = model.embed_dim # Assuming it's directly accessible
    model.output = nn.Identity()
    return model, embed_dim

def _create_causal_mask(seq_len: int, device: torch.device):
    # Creates a lower triangular mask for causal attention.
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """Indexes the causal mask based on input positions."""
    # Ensure mask and input_pos are compatible.
    # This implementation assumes input_pos contains indices valid for the mask's first dimension.
    try:
        return mask[input_pos]
    except IndexError as e:
        logger.error(f"IndexError in _index_causal_mask: mask shape {mask.shape}, input_pos shape {input_pos.shape}, input_pos max {input_pos.max() if input_pos.numel() > 0 else 'N/A'}. Error: {e}")
        raise

def _multinomial_sample_one_no_sync(probs):
    # Efficient multinomial sampling.
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    # Samples from the top-k logits.
    logits = logits / temperature
    filter_value: float = -float("Inf")
    # Ensure topk is valid
    k = min(topk, logits.size(-1))
    if k <= 0:
        k = 1 # Sample at least the top token
    top_k_logits, _ = torch.topk(logits, k)
    min_top_k_logit = top_k_logits[..., -1, None] # Get the k-th logit value

    indices_to_remove = logits < min_top_k_logit
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    # Softmax calculation might be numerically unstable if all values are -inf after filtering
    # Adding a check here
    if torch.all(scores_processed == filter_value):
        logger.warning("All logits filtered out in sample_topk. Returning argmax of original logits.")
        # Fallback: return the single most likely token from the original distribution
        return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.int)

    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token

@dataclass
class ModelArgs:
    backbone_flavor: str = "llama-1B"
    decoder_flavor: str = "llama-100M"
    text_vocab_size: int = 128256
    audio_vocab_size: int = 2051 # CORRECTED based on state_dict mismatch errors
    audio_num_codebooks: int = 32

# We might need PyTorchModelHubMixin if the loading relies on it, but prioritize local loading.
# If local loading works without it, it can be removed.
try:
    from huggingface_hub import PyTorchModelHubMixin
    BaseModelClass = nn.Module, PyTorchModelHubMixin
except ImportError:
    logger.warning("huggingface_hub not found. PyTorchModelHubMixin features unavailable.")
    BaseModelClass = nn.Module, # Inherit only from nn.Module

# The Model class from the original file
class Model(*BaseModelClass):
    # Add repo_url etc. only if PyTorchModelHubMixin is used
    # repo_url="https://github.com/SesameAILabs/csm", pipeline_tag="text-to-speech", license="apache-2.0"
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        logger.info(f"Initializing Model with config: {config}")
        logger.info(f"Initializing backbone: {config.backbone_flavor}")
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())

        logger.info(f"Initializing decoder: {config.decoder_flavor}")
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        logger.info("Initializing embeddings and heads...")
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        # Calculate total embedding size for audio based on codebooks
        audio_total_embeddings = config.audio_vocab_size * config.audio_num_codebooks
        self.audio_embeddings = nn.Embedding(audio_total_embeddings, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        # Initialize audio_head parameter
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))
        # Optional: Initialize weights (e.g., Kaiming uniform) if needed, otherwise they load from state_dict
        # nn.init.kaiming_uniform_(self.audio_head, a=math.sqrt(5))
        logger.info("Model components initialized.")


    def setup_caches(self, max_batch_size: int) -> None:
        """Setup KV caches and causal masks."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        logger.info(f"Setting up caches for batch_size={max_batch_size}, dtype={dtype}, device={device}")
        with torch.device(device):
            # --- Backbone Cache Setup (Keep as corrected before) ---
            if hasattr(self.backbone, 'setup_caches'):
                 try:
                      self.backbone.setup_caches(batch_size=max_batch_size, dtype=dtype)
                      logger.info("Backbone caches setup called with batch_size and dtype.")
                 except TypeError:
                     try:
                         self.backbone.setup_caches(max_batch_size, dtype=dtype)
                         logger.info("Backbone caches setup called positionally with batch_size and dtype.")
                     except TypeError as e:
                          logger.error(f"Failed to call backbone.setup_caches: {e}")
                          raise
            else:
                 logger.warning("Backbone model does not have setup_caches method.")

            # --- Decoder Cache Setup (MODIFIED) ---
            if hasattr(self.decoder, 'setup_caches'):
                 try:
                     # Try calling with ONLY batch_size and dtype
                     self.decoder.setup_caches(batch_size=max_batch_size, dtype=dtype)
                     logger.info(f"Decoder caches setup called with batch_size and dtype.")
                 except TypeError:
                      try:
                          # Try positional ONLY with batch_size and dtype
                          self.decoder.setup_caches(max_batch_size, dtype=dtype)
                          logger.info(f"Decoder caches setup called positionally with batch_size and dtype.")
                      except TypeError as e:
                           logger.error(f"Failed to call decoder.setup_caches with batch_size and dtype: {e}")
                           # If it fails here, the signature is very unusual
                           raise
            else:
                 logger.warning("Decoder model does not have setup_caches method.")
            # --- End Modifications ---

        # Register buffers for masks (keep as is)
        backbone_max_seq_len = getattr(self.backbone, 'max_seq_len', 2048)
        self.register_buffer("backbone_causal_mask", _create_causal_mask(backbone_max_seq_len, device), persistent=False)
        decoder_max_seq_len = self.config.audio_num_codebooks
        self.register_buffer("decoder_causal_mask", _create_causal_mask(decoder_max_seq_len, device), persistent=False)
        logger.info("Causal masks registered.")

    # --- generate_frame (as provided in original) ---
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """Generates one frame of audio tokens."""
        dtype = next(self.parameters()).dtype
        # Ensure caches are enabled if the model uses them
        # assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"

        # Get current mask slice based on positions
        # Ensure input_pos are valid indices for backbone_causal_mask
        max_pos = input_pos.max()
        if max_pos >= self.backbone_causal_mask.shape[0]:
             logger.error(f"input_pos max index ({max_pos}) out of bounds for backbone_causal_mask shape {self.backbone_causal_mask.shape}")
             # Handle error appropriately, maybe raise or return error indicator
             raise IndexError("input_pos out of bounds for backbone mask")
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)

        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1) # Apply mask
        # Sum embeddings across the codebook+text dimension
        h = masked_embeds.sum(dim=2)

        # Pass through backbone
        # Handle models with/without cache enabled status check
        backbone_output = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        h = backbone_output.to(dtype=dtype)


        # Process last hidden state for codebook 0
        last_h = h[:, -1, :] # Get hidden state of the last token in the sequence
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample) # Get embedding for the sampled token

        # Prepare input for the decoder loop
        # Input includes the backbone's last hidden state and the embedding of the first sampled codebook token
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone() # Start accumulating sampled tokens
        # Decoder positions start from 0 for the first input (last_h)
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Reset decoder caches for each new frame generation
        if hasattr(self.decoder, 'reset_caches'):
            self.decoder.reset_caches()

        # Loop through remaining codebooks (1 to N-1)
        for i in range(1, self.config.audio_num_codebooks):
            # Get the mask for the current decoder step
            max_decoder_pos = curr_pos.max()
            if max_decoder_pos >= self.decoder_causal_mask.shape[0]:
                 logger.error(f"curr_pos max index ({max_decoder_pos}) out of bounds for decoder_causal_mask shape {self.decoder_causal_mask.shape}")
                 raise IndexError("curr_pos out of bounds for decoder mask")
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)

            # Pass projected hidden state through the decoder
            # The input to the decoder is the embedding from the *previous* step (or last_h initially)
            decoder_input = self.projection(curr_h)
            decoder_output = self.decoder(decoder_input, input_pos=curr_pos, mask=curr_decoder_mask)
            decoder_h = decoder_output.to(dtype=dtype)

            # Get logits for the current codebook using the specific head
            # Use the hidden state corresponding to the *last* input token for prediction
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample) # Embed the newly sampled token

            # Update the input for the *next* decoder step
            curr_h = ci_embed # Input for next step is the embedding of the token just sampled
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1) # Add sample to the frame
            # Increment position for the next step
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample # Return the complete frame of sampled tokens

    # --- reset_caches (as provided) ---
    def reset_caches(self):
        if hasattr(self.backbone, 'reset_caches'): self.backbone.reset_caches()
        if hasattr(self.decoder, 'reset_caches'): self.decoder.reset_caches()
        logger.debug("Model caches reset.")


    # --- Embedding helpers (as provided) ---
    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds audio tokens using offsets for each codebook."""
        offset = codebook * self.config.audio_vocab_size
        return self.audio_embeddings(tokens + offset)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds combined audio and text tokens."""
        # tokens shape: (batch_size, seq_len, audio_num_codebooks + 1)
        # Last element is text token, others are audio tokens for each codebook

        # Embed text token (index -1)
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2) # Add dimension for concatenation

        # Embed audio tokens (indices 0 to N-1)
        audio_tokens = tokens[:, :, :-1] # Select audio codebook tokens
        # Apply offsets for each codebook
        codebook_indices = torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        audio_tokens_with_offset = audio_tokens + (self.config.audio_vocab_size * codebook_indices)

        # Embed audio tokens
        # Reshape might be needed if embedding layer expects flat input
        b, s, cb = audio_tokens_with_offset.shape
        audio_embeds = self.audio_embeddings(audio_tokens_with_offset.view(b * s, cb))
        audio_embeds = audio_embeds.view(b, s, cb, -1) # Reshape back: (batch, seq, codebooks, dim)

        # Concatenate audio and text embeddings along the codebook/text dimension
        return torch.cat([audio_embeds, text_embeds], dim=2) # Result shape: (b, s, cb+1, dim)

    @classmethod
    def from_local_pretrained(cls, local_path: str, device: str = 'cpu'):
        """Loads the model configuration and weights from a local directory."""
        local_path = Path(local_path)
        logger.info(f"Loading model from local path: {local_path}")

        # 1. Config (keep as is)
        config = ModelArgs()
        logger.info(f"Using ModelArgs: {config}")

        # 2. Initialize model (keep as is)
        model = cls(config)

        # 3. Load the state dictionary (weights) - **MODIFIED**
        potential_weight_files = ["model.safetensors", "pytorch_model.bin", "csm_weights.pt", "csm-1b.pt"] # Prioritize safetensors
        weight_file_path = None
        weight_file_type = None

        for filename in potential_weight_files:
            path_to_check = local_path / filename
            if path_to_check.is_file():
                weight_file_path = path_to_check
                if filename.endswith(".safetensors"):
                    weight_file_type = "safetensors"
                else:
                    weight_file_type = "pytorch"
                logger.info(f"Found weight file: {weight_file_path} (Type: {weight_file_type})")
                break

        if weight_file_path is None:
            raise FileNotFoundError(f"Could not find model weight file in {local_path}")

        logger.info(f"Loading weights from {weight_file_path} using {weight_file_type} loader...")

        # --- Load based on file type ---
        if weight_file_type == "safetensors":
            # Use safetensors loader
            try:
                 # load_safetensors_file loads directly to the specified device
                 state_dict = load_safetensors_file(weight_file_path, device=device)
                 logger.info(f"Safetensors file loaded successfully to device '{device}'.")
            except Exception as e:
                 logger.error(f"Failed to load safetensors file {weight_file_path}: {e}", exc_info=True)
                 raise
        else: # Assume PyTorch format (.bin, .pt)
            try:
                 # Use torch.load with weights_only=True for security if possible
                 # map_location ensures loading onto the target device
                 state_dict = torch.load(weight_file_path, map_location=torch.device(device), weights_only=True)
                 logger.info(f"PyTorch file loaded successfully with weights_only=True to device '{device}'.")
            except (_pickle.UnpicklingError, RuntimeError, EOFError) as load_err:
                 # Fallback for older files or potential corruption, without weights_only
                 logger.warning(f"torch.load with weights_only=True failed ({load_err}). Retrying with weights_only=False (potential security risk).")
                 try:
                     state_dict = torch.load(weight_file_path, map_location=torch.device(device), weights_only=False) # Potential security risk
                     logger.info(f"PyTorch file loaded successfully with weights_only=False to device '{device}'.")
                 except Exception as e:
                     logger.error(f"Failed to load PyTorch file {weight_file_path} even with weights_only=False: {e}", exc_info=True)
                     raise
            except Exception as e:
                 logger.error(f"Failed to load PyTorch file {weight_file_path}: {e}", exc_info=True)
                 raise


        # 4. Load state dict into model
        logger.info("Loading state dict into model...")
        try:
             load_result = model.load_state_dict(state_dict, strict=True)
             logger.info(f"State dict loaded successfully (strict=True). Result: {load_result}")
        except RuntimeError as e:
             # Log only the summary error from strict loading
             logger.error(f"Error loading state dict (strict=True): Mismatched keys found. Details: {str(e)[:500]}...")
             logger.info("Attempting to load state dict with strict=False...")
             try:
                  load_result = model.load_state_dict(state_dict, strict=False)
                  logger.warning(f"State dict loaded with strict=False. Result: {load_result}")
                  logger.warning("Mismatched/missing keys were ignored. Check error log above if issues arise.")
             except Exception as e2:
                  logger.error(f"Failed to load state dict even with strict=False. Error: {e2}")
                  raise RuntimeError(f"Failed to load state dict from {weight_file_path}. Check logs for mismatch details.") from e2
        except Exception as e_other:
             logger.error(f"An unexpected error occurred during state dict loading: {e_other}", exc_info=True)
             raise

        model.to(device)
        model.eval()
        logger.info(f"Model ready on device '{device}' and set to eval mode.")
        return model