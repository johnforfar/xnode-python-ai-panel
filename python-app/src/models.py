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

    # --- RESTORE _index_causal_mask from OLD version ---
    def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
        """
        Indexes a square causal mask using a tensor of indices.
        Args:
            mask: (max_seq_len, max_seq_len)
            input_pos: (batch_size, seq_len)
        Returns:
            (batch_size, seq_len, max_seq_len)
        """
        logger.debug(f"--- _index_causal_mask (Old Version) ---")
        logger.debug(f"  Input mask shape: {mask.shape}")
        logger.debug(f"  Input indices (input_pos) shape: {input_pos.shape}")
        try:
            # Select rows based on input_pos
            r = mask[input_pos, :]
            logger.debug(f"  Output mask shape: {r.shape}")
            return r
        except IndexError as e:
            logger.error(f"  IndexError in _index_causal_mask: {e}", exc_info=True)
            logger.error(f"  Max index requested: {input_pos.max().item() if input_pos.numel() > 0 else 'N/A'}")
            logger.error(f"  Mask dimension size: {mask.shape[0]}")
            raise
        except Exception as e:
            logger.error(f"  Unexpected error in _index_causal_mask: {e}", exc_info=True)
            raise
    # --- End Restore ---

    def setup_caches(self, max_batch_size: int) -> None:
        """Setup KV caches and causal masks."""
        device = next(self.parameters()).device
        self.logger.info(f"Setting up caches for batch_size={max_batch_size}, device={device}")

        with torch.device(device):
            # Setup backbone cache - try without dtype
            if hasattr(self.backbone, 'setup_caches'):
                 try:
                      self.backbone.setup_caches(batch_size=max_batch_size)
                      self.logger.info(">>> Backbone caches setup call completed.")
                 except TypeError as e:
                      self.logger.error(f"Failed to call backbone.setup_caches: {e}")
                      # Try original call with dtype as fallback
                      try:
                          dtype = next(self.parameters()).dtype
                          self.logger.warning("Retrying backbone setup_caches WITH dtype...")
                          self.backbone.setup_caches(batch_size=max_batch_size, dtype=dtype)
                          self.logger.info(">>> Backbone caches setup call completed (with dtype fallback).")
                      except Exception as e2:
                          self.logger.error(f"Fallback backbone setup_caches also failed: {e2}")
                          raise
                 except Exception as e_other:
                     self.logger.error(f"Unexpected error during backbone cache setup: {e_other}")
                     raise
            else:
                 self.logger.warning("Backbone does not have setup_caches method.")

            # Setup decoder cache - try without dtype
            if hasattr(self.decoder, 'setup_caches'):
                 try:
                     self.decoder.setup_caches(batch_size=max_batch_size)
                     self.logger.info(f">>> Decoder caches setup call completed.")
                 except TypeError as e:
                     self.logger.error(f"Failed to call decoder.setup_caches: {e}")
                     # Try original call with dtype as fallback
                     try:
                          dtype = next(self.parameters()).dtype
                          self.logger.warning("Retrying decoder setup_caches WITH dtype...")
                          self.decoder.setup_caches(batch_size=max_batch_size, dtype=dtype)
                          self.logger.info(f">>> Decoder caches setup call completed (with dtype fallback).")
                     except Exception as e2:
                         self.logger.error(f"Fallback decoder setup_caches also failed: {e2}")
                         raise
                 except Exception as e_other:
                     self.logger.error(f"Unexpected error during decoder cache setup: {e_other}")
                     raise
            else:
                self.logger.warning("Decoder does not have setup_caches method.")

        # Register buffers for masks AFTER cache setup
        backbone_max_seq_len = getattr(self.backbone, 'max_seq_len', 2048) # Get actual max_seq_len if possible
        self.register_buffer("backbone_causal_mask", _create_causal_mask(backbone_max_seq_len, device), persistent=False)
        self.register_buffer("decoder_causal_mask", _create_causal_mask(decoder_max_seq_len, device), persistent=False)
        self.logger.info("Causal mask buffers registered.")

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
        input_pos: torch.Tensor, # Initial input_pos for the backbone
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """Generates one frame of audio tokens (KV Caching ENABLED)."""
        # --- Start of Changes ---
        # MODIFY Assertions to check property instead of calling method
        # Default to False if the attribute doesn't exist
        backbone_caches_enabled = getattr(self.backbone, 'caches_are_enabled', False)
        decoder_caches_enabled = getattr(self.decoder, 'caches_are_enabled', False)

        if not backbone_caches_enabled:
             self.logger.error("Assertion failed: Backbone caches property 'caches_are_enabled' is False or missing!")
             # Log relevant attributes of self.backbone to help debug
             self.logger.error(f"Backbone attributes: {dir(self.backbone)}")
             raise AssertionError("Backbone caches are not enabled. Call setup_caches first.")
        if not decoder_caches_enabled:
             self.logger.error("Assertion failed: Decoder caches property 'caches_are_enabled' is False or missing!")
             # Log relevant attributes of self.decoder
             self.logger.error(f"Decoder attributes: {dir(self.decoder)}")
             raise AssertionError("Decoder caches are not enabled. Call setup_caches first.")
        self.logger.debug("Assertions passed: Cache properties 'caches_are_enabled' are True.")
        # --- End of Changes ---

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # --- Backbone Section (Keep restored logic) ---
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        self.logger.debug(f"--- Backbone Call (Cache Enabled) ---")
        self.logger.debug(f"  Input h shape: {h.shape}")
        self.logger.debug(f"  Input input_pos shape: {input_pos.shape}, values:\n{input_pos}")

        try:
            # Calculate mask using the OLD restored _index_causal_mask
            curr_backbone_mask = self._index_causal_mask(self.backbone_causal_mask, input_pos)
            self.logger.debug(f"  Calculated curr_backbone_mask shape: {curr_backbone_mask.shape}")
        except Exception as e:
            self.logger.error(f"Error indexing backbone causal mask: {e}", exc_info=True)
            raise ValueError(f"Failed to create backbone mask: {e}") from e

        try:
            # Call backbone WITH input_pos and the calculated mask
            backbone_output = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
            self.logger.debug(f"  Backbone output shape: {backbone_output.shape}")
            h = backbone_output.to(dtype=dtype)
        except Exception as e:
             self.logger.error(f"--- ERROR during self.backbone call (Cache Enabled) ---")
             self.logger.error(f"  Exception: {e}", exc_info=False)
             self.logger.error(f"  Input h shape: {h.shape}")
             self.logger.error(f"  Input input_pos shape: {input_pos.shape}")
             self.logger.error(f"  Input curr_backbone_mask shape: {curr_backbone_mask.shape if curr_backbone_mask is not None else 'None'}")
             raise
        # --- End Backbone Section ---

        # --- Decoder Section (Keep restored logic) ---
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Check if decoder_causal_mask exists (important after restoring setup_caches)
        if not hasattr(self, 'decoder_causal_mask'):
             self.logger.error("self.decoder_causal_mask not found after setup_caches!")
             raise AttributeError("Model is missing 'decoder_causal_mask', cannot generate.")

        # Decoder loop
        for i in range(1, self.config.audio_num_codebooks):
            try:
                # --- ADD Decoder Cache Reset (from old version) ---
                if hasattr(self.decoder, 'reset_caches'):
                    self.decoder.reset_caches()
                # --- End Add ---

                decoder_input = self.projection(curr_h)
                try:
                    # Calculate mask using the OLD restored _index_causal_mask
                    curr_decoder_mask = self._index_causal_mask(self.decoder_causal_mask, curr_pos)
                except Exception as e:
                    self.logger.error(f"Error indexing decoder causal mask at step {i}: {e}", exc_info=True)
                    raise ValueError(f"Failed to create decoder mask at step {i}: {e}") from e

                self.logger.debug(f"--- Decoder Step {i} (Cache Enabled) ---")
                self.logger.debug(f"  Input decoder_input shape: {decoder_input.shape}")
                self.logger.debug(f"  Input curr_pos shape: {curr_pos.shape}, values:\n{curr_pos}")
                self.logger.debug(f"  Calculated curr_decoder_mask shape: {curr_decoder_mask.shape}")

                # Call decoder WITH input_pos and the calculated mask
                decoder_output = self.decoder(decoder_input, input_pos=curr_pos, mask=curr_decoder_mask)

                decoder_h = decoder_output.to(dtype=dtype)
                self.logger.debug(f"  Output decoder_h shape: {decoder_h.shape}")

            except Exception as e:
                self.logger.error(f"--- ERROR during self.decoder call at step {i} (Cache Enabled) ---")
                self.logger.error(f"  Exception: {e}", exc_info=False)
                self.logger.error(f"  Input decoder_input shape: {decoder_input.shape}")
                self.logger.error(f"  Input curr_pos shape: {curr_pos.shape}, values:\n{curr_pos}")
                self.logger.error(f"  Input curr_decoder_mask shape: {curr_decoder_mask.shape if curr_decoder_mask is not None else 'None'}")
                raise

            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1
        # --- End Decoder Section ---

        return curr_sample

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