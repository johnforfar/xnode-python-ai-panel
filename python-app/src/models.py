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
    if torch.cuda.is_available():
        model.to("cuda")
    return model, embed_dim

def _create_causal_mask(seq_len: int, device: torch.device):
    # Creates a lower triangular mask for causal attention.
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
    # Add logging if needed for debugging mask shapes/indices
    # logger.debug(f"Indexing mask {mask.shape} with indices {input_pos.shape}")
    try:
        r = mask[input_pos, :]
        return r
    except IndexError as e:
         # logger.error(f"IndexError in _index_causal_mask: {e}", exc_info=True)
         # logger.error(f"Max index: {input_pos.max().item()}, Mask dim: {mask.shape[0]}")
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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}") # Add logger to class instance if needed

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
        logger.info("Model components initialized (KV Caching ENABLED).")

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype) -> None:
        """Setup KV caches."""
        device = next(self.parameters()).device
        logger.info(f"Setting up caches for batch_size={max_batch_size}, device={device}, dtype={dtype}")

        try:
            logger.debug("Calling backbone.setup_caches...")
            self.backbone.setup_caches(max_batch_size, dtype=dtype)
            logger.debug(">>> Backbone caches setup call completed.")
        except Exception as e:
            logger.error(f"Failed to call backbone.setup_caches: {e}", exc_info=True)
            # Attempt fallback if specific error occurs? (Removed retry from here, handled in generator init)

        try:
            logger.debug("Calling decoder.setup_caches...")
            self.decoder.setup_caches(max_batch_size, dtype=dtype, decoder_max_seq_len=self.config.audio_num_codebooks)
            logger.debug(">>> Decoder caches setup call completed.")
        except Exception as e:
            logger.error(f"Failed to call decoder.setup_caches: {e}", exc_info=True)
            # Attempt fallback if specific error occurs? (Removed retry from here, handled in generator init)

        # Causal mask setup (needs correct sequence lengths)
        try:
            backbone_max_seq_len = self.backbone.max_seq_len
            decoder_mask_len = self.config.audio_num_codebooks
            logger.debug(f"Creating causal masks: backbone_len={backbone_max_seq_len}, decoder_len={decoder_mask_len}")

            # Detach previous buffers if they exist to allow replacement
            if hasattr(self, 'backbone_causal_mask'):
                del self.backbone_causal_mask
            if hasattr(self, 'decoder_causal_mask'):
                del self.decoder_causal_mask

            self.register_buffer("backbone_causal_mask", _create_causal_mask(backbone_max_seq_len, device), persistent=False)
            self.register_buffer("decoder_causal_mask", _create_causal_mask(decoder_mask_len, device), persistent=False)
            logger.debug("Causal masks registered successfully.")
        except AttributeError as e:
             logger.error(f"Failed to get max_seq_len from backbone: {e}. Cannot create causal masks.")
        except Exception as e:
             logger.error(f"Error creating causal masks: {e}", exc_info=True)

    def reset_caches(self):
        if hasattr(self.backbone, 'reset_caches'):
            self.backbone.reset_caches()
            # self.logger.debug("Backbone cache reset.") # Optional debug log
        if hasattr(self.decoder, 'reset_caches'):
            self.decoder.reset_caches()
            # self.logger.debug("Decoder cache reset.") # Optional debug log

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor, # Global input_pos for the backbone
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """Generates one frame of audio tokens (KV Caching ENABLED)."""
        # Based on ORIGINAL Sesame models.py logic
        backbone_caches_enabled = getattr(self.backbone, 'caches_are_enabled', False)
        decoder_caches_enabled = getattr(self.decoder, 'caches_are_enabled', False)

        if not backbone_caches_enabled:
             self.logger.error("Assertion failed: Backbone caches not enabled!")
             raise AssertionError("Backbone caches are not enabled. Call setup_caches first.")
        if not decoder_caches_enabled:
             self.logger.error("Assertion failed: Decoder caches not enabled!")
             raise AssertionError("Decoder caches are not enabled. Call setup_caches first.")
        # self.logger.debug("Assertions passed: Caches enabled.") # Optional

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # --- Backbone Section (As before) ---
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        # self.logger.debug(f"--- Backbone Call ---")
        try:
            curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
            backbone_output = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
            h = backbone_output # Use output directly (CPU float32 fix)
        except Exception as e:
             self.logger.error(f"--- ERROR during self.backbone call ---", exc_info=True)
             raise
        # --- End Backbone Section ---

        # --- Decoder Section (Reverted to Original Sesame Logic) ---
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1) # Input for first decoder step
        curr_sample = c0_sample.clone() # Stores generated codebook tokens
        # Position tracker for the decoder input sequence (starts at 0, grows)
        curr_pos_decoder = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        if not hasattr(self, 'decoder_causal_mask'):
             self.logger.error("self.decoder_causal_mask not found!")
             raise AttributeError("Model is missing 'decoder_causal_mask', cannot generate.")

        # Decoder loop: i goes from 1 to 31 (predicting codebooks 1 to 31)
        # --- REVERT decoder loop logic to match original ---
        if hasattr(self.decoder, 'reset_caches'): # Reset cache before the loop
             self.decoder.reset_caches()
        else:
             self.logger.warning("Decoder has no reset_caches method!")

        for i in range(1, self.config.audio_num_codebooks):
            # Reset cache *inside* the loop (as per original models.py)
            # This seems necessary for the original non-streaming logic to work with the cache
            # if hasattr(self.decoder, 'reset_caches'):
            #    self.decoder.reset_caches() # Keep reset inside loop as per original

            try:
                # Project the current hidden state (curr_h)
                projected_input = self.projection(curr_h) # Input for decoder
                # Index the mask using the current decoder position tracker
                curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos_decoder)

                # self.logger.debug(f"--- Decoder Step {i} ---")
                # self.logger.debug(f"  Input projected_input shape: {projected_input.shape}")
                # self.logger.debug(f"  Input curr_pos_decoder shape: {curr_pos_decoder.shape}, values:\n{curr_pos_decoder}")
                # self.logger.debug(f"  Calculated curr_decoder_mask shape: {curr_decoder_mask.shape}")

                # Call decoder with the projected input, its positions, and mask
                decoder_output = self.decoder(projected_input, input_pos=curr_pos_decoder, mask=curr_decoder_mask)
                decoder_h = decoder_output # Use output directly (CPU float32 fix)

            except Exception as e:
                self.logger.error(f"--- ERROR during self.decoder call at step {i} ---", exc_info=True)
                raise

            # Sample the next codebook token (ci)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            # Update state for the *next* iteration:
            curr_h = ci_embed # Input for next step is the embedding of the token just generated
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos_decoder = curr_pos_decoder[:, -1:] + 1 # Increment decoder position tracker
        # --- End Revert ---
        # --- End Decoder Section ---

        return curr_sample # Shape [B, 32]

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
    def from_local_pretrained(cls, local_path: str):
        """Loads the model configuration and weights from a local directory."""
        local_path = Path(local_path)
        logger.info(f"Loading model from local path: {local_path}")

        # 1. Config (keep as is)
        config = ModelArgs()
        logger.info(f"Using ModelArgs: {config}")

        # 2. Initialize model (keep as is)
        model = cls(config)

        # 3. Load the state dictionary (weights)
        potential_weight_files = ["model.safetensors", "pytorch_model.bin", "csm_weights.pt", "csm-1b.pt"]
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

        # Determine map_location for torch.load (use 'cpu' by default for initial load)
        # Loading directly to CUDA can sometimes cause issues if tensors were saved from CPU
        map_location_device = 'cpu' # Load to CPU first for stability
        logger.debug(f"Using map_location='{map_location_device}' for torch.load")

        if weight_file_type == "safetensors":
            try:
                 # load_safetensors_file loads directly to the specified device. Let's try CPU first.
                 state_dict = load_safetensors_file(weight_file_path, device=map_location_device)
                 logger.info(f"Safetensors file loaded successfully to device '{map_location_device}'.")
            except Exception as e:
                 logger.error(f"Failed to load safetensors file {weight_file_path}: {e}", exc_info=True)
                 raise
        else: # PyTorch format
            try:
                 state_dict = torch.load(weight_file_path, map_location=map_location_device, weights_only=True)
                 logger.info(f"PyTorch file loaded successfully with weights_only=True to device '{map_location_device}'.")
            except (pickle.UnpicklingError, RuntimeError, EOFError) as load_err:
                 logger.warning(f"torch.load with weights_only=True failed ({load_err}). Retrying with weights_only=False (potential security risk).")
                 try:
                     state_dict = torch.load(weight_file_path, map_location=map_location_device, weights_only=False)
                     logger.info(f"PyTorch file loaded successfully with weights_only=False to device '{map_location_device}'.")
                 except Exception as e:
                     logger.error(f"Failed to load PyTorch file {weight_file_path} even with weights_only=False: {e}", exc_info=True)
                     raise
            except Exception as e:
                 logger.error(f"Failed to load PyTorch file {weight_file_path}: {e}", exc_info=True)
                 raise

        # 4. Load state dict into model (model is currently on CPU)
        logger.info("Loading state dict into model...")
        try:
             load_result = model.load_state_dict(state_dict, strict=True)
             logger.info(f"State dict loaded successfully (strict=True). Result: {load_result}")
        except RuntimeError as e:
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

        model.eval()
        logger.info(f"Model loaded to CPU and set to eval mode. Device transfer happens after.")
        return model