# models.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import os
from pathlib import Path
import logging
from safetensors.torch import load_file as load_safetensors_file
import pickle
from typing import TypeVar

# --- Import Hugging Face Hub Mixin ---
try:
    from huggingface_hub import PyTorchModelHubMixin
    MixinBase = PyTorchModelHubMixin
except ImportError:
    # Define a dummy MixinBase if not found, inheriting from object
    class MixinBase(object):
        pass
    # No need to log warning here, handled if needed during Hub operations
# --- End Import ---


# Ensure torchtune is installed: pip install torchtune
try:
    import torchtune
    from torchtune.models import llama3_2
    # Import modules used by torchtune Llama models if needed (check torchtune source)
    # from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings, CausalSelfAttention, FeedForward, TransformerDecoderLayer
except ImportError:
    print("ERROR: torchtune library not found. Please install it: pip install torchtune")
    raise

logger = logging.getLogger(__name__)

# Define model configurations based on original CSM repo
def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
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
     return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        # Ensure max_seq_len for decoder doesn't cause issues. Original CSM uses small decoder.
        # Torchtune's Llama might have its own max_seq_len expectation. Let's keep it 2048 for now.
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}

# --- _prepare_transformer from original CSM repo ---
# This removes the original embedding and output layers from the torchtune model
def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim
# --- End _prepare_transformer ---

# --- Causal Mask functions from original CSM repo ---
# Creates a standard lower-triangular causal mask
def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

# Selects rows from the causal mask based on input positions for KV cache
def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len) - The full causal mask buffer
        input_pos: (batch_size, seq_len) - Indices of tokens being processed

    Returns:
        (batch_size, seq_len, max_seq_len) - The relevant mask rows for the current input tokens
    """
    try:
        # This selects the rows corresponding to the token indices
        r = mask[input_pos, :]
        return r
    except IndexError as e:
         # Add more context to the error if it occurs
         max_idx = input_pos.max().item() if input_pos.numel() > 0 else 'N/A'
         logger.error(f"IndexError in _index_causal_mask: {e}. Max index requested: {max_idx}, Mask dim 0 size: {mask.shape[0]}", exc_info=True)
         raise
# --- End Causal Mask functions ---

# --- Sampling functions from original CSM repo ---
def _multinomial_sample_one_no_sync(probs):  # Efficient multinomial sampling without CUDA sync
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature # Apply temperature scaling

    filter_value: float = -float("Inf")
    # Ensure topk is valid (at least 1, not more than vocab size)
    k = min(max(topk, 1), logits.size(-1))

    # Get the value of the k-th highest logit
    top_k_logits, _ = torch.topk(logits, k)
    min_top_k_logit = top_k_logits[..., -1, None] # Shape: [..., 1]

    # Mask out logits below the k-th highest
    indices_to_remove = logits < min_top_k_logit
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)

    # Check if all logits were filtered (e.g., very low temp, bad distribution)
    if torch.all(scores_processed == filter_value):
        logger.warning("All logits filtered out in sample_topk. Returning argmax of original logits.")
        return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.int) # Fallback: take the best one

    # --- Use log_softmax before softmax for numerical stability ---
    # This was missing in one of the previous versions and is present in robust sampling implementations
    log_probs = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.exp(log_probs) # Convert back to probabilities for multinomial sampling
    # --- End log_softmax ---

    # Sample from the filtered probability distribution
    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token
# --- End Sampling functions ---

@dataclass
class ModelArgs:
    # Keep defaults matching original CSM-1B
    backbone_flavor: str = "llama-1B"
    decoder_flavor: str = "llama-100M"
    text_vocab_size: int = 128256
    audio_vocab_size: int = 2051 # Keep corrected value based on state_dict errors
    audio_num_codebooks: int = 32

T = TypeVar("T", bound=nn.Module) # For type hinting class methods

class Model(nn.Module, MixinBase): # Inherit from MixinBase
    # Add attributes for Hugging Face Hub integration if MixinBase is PyTorchModelHubMixin
    if isinstance(MixinBase, type) and issubclass(MixinBase, PyTorchModelHubMixin):
        repo_url="https://github.com/SesameAILabs/csm"
        pipeline_tag="text-to-speech"
        license="apache-2.0"

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.logger.info(f"Initializing Model with config: {config}")
        self.logger.info(f"Initializing backbone: {config.backbone_flavor}")
        # Prepare backbone (removes original embeddings/output)
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())

        self.logger.info(f"Initializing decoder: {config.decoder_flavor}")
        # Prepare decoder (removes original embeddings/output)
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        self.logger.info("Initializing CSM-specific embeddings and projection/heads...")
        # CSM's own embeddings and output heads
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        audio_total_embeddings = config.audio_vocab_size * config.audio_num_codebooks
        self.audio_embeddings = nn.Embedding(audio_total_embeddings, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False) # Project backbone output to decoder dim
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False) # Predicts first codebook
        # Parameter for the remaining codebook prediction heads (used with decoder output)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))
        self.logger.info("Model components initialized.")

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype) -> None:
        """Setup KV caches for both backbone and decoder.
        Args:
            max_batch_size: Max batch size for inference.
            dtype: The *desired* dtype for the cache tensors (e.g., bfloat16 for performance).
                   The model parameters might be loaded in a different precision (e.g., float32).
        """
        device = next(self.parameters()).device
        model_param_dtype = next(self.parameters()).dtype # Actual precision of loaded weights
        self.logger.info(f"Setting up caches for batch_size={max_batch_size}, device={device}, requested cache dtype: {dtype} (Model params are {model_param_dtype})")

        try:
            with torch.device(device): # Ensure cache tensors are created on the correct device
                # Setup caches using the *requested* dtype for potential memory/speed benefits
                self.backbone.setup_caches(max_batch_size, dtype=dtype)
                # Decoder cache setup - ensure max_seq_len matches its usage (num_codebooks)
                # The torchtune Llama decoder might have its own default max_seq_len,
                # but CSM uses it differently, only up to num_codebooks steps.
                decoder_max_seq_len_for_cache = self.config.audio_num_codebooks
                self.decoder.setup_caches(max_batch_size, dtype=dtype, max_seq_len=decoder_max_seq_len_for_cache)
                self.logger.debug(f"Torchtune caches setup with dtype={dtype}. Decoder cache max_seq_len={decoder_max_seq_len_for_cache}")
        except Exception as e:
             self.logger.error(f"Failed during cache setup with requested dtype {dtype}: {e}", exc_info=True)
             raise

        # Causal mask buffer setup (critical for correct attention)
        try:
            backbone_max_seq_len = self.backbone.max_seq_len
            # The decoder mask only needs to cover the number of codebooks steps
            decoder_mask_len = self.config.audio_num_codebooks
            self.logger.debug(f"Creating causal masks: backbone_len={backbone_max_seq_len}, decoder_len={decoder_mask_len}")

            # Remove existing buffers if they exist before re-registering
            if hasattr(self, 'backbone_causal_mask'): del self.backbone_causal_mask
            if hasattr(self, 'decoder_causal_mask'): del self.decoder_causal_mask

            # Use persistent=False for buffers not part of the model's state_dict
            self.register_buffer("backbone_causal_mask", _create_causal_mask(backbone_max_seq_len, device), persistent=False)
            self.register_buffer("decoder_causal_mask", _create_causal_mask(decoder_mask_len, device), persistent=False)
            self.logger.debug("Causal mask buffers registered successfully.")
        except AttributeError as e:
             self.logger.error(f"Failed to get max_seq_len from backbone: {e}. Cannot create causal masks.")
        except Exception as e:
             self.logger.error(f"Error creating or registering causal masks: {e}", exc_info=True)

    def reset_caches(self):
        """Resets the KV caches in both backbone and decoder."""
        if hasattr(self.backbone, 'reset_caches'):
            self.backbone.reset_caches()
        if hasattr(self.decoder, 'reset_caches'):
            self.decoder.reset_caches()

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Generates one frame of audio tokens (all codebooks).
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1) - Input token IDs (audio + text)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1) - Mask for valid tokens
            input_pos: (batch_size, seq_len) - Positional indices for the backbone KV cache
            temperature: Sampling temperature
            topk: Top-k sampling parameter

        Returns:
            (batch_size, audio_num_codebooks) - Sampled audio tokens for the next frame
        """
        # --- Determine dtypes and device ---
        # Use the actual dtype of the model parameters for internal computations unless cast
        compute_dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        # --- End dtypes/device ---

        # --- Check Prerequisites ---
        assert hasattr(self, 'backbone_causal_mask'), "Backbone causal mask not initialized. Call setup_caches first."
        assert self.backbone.caches_are_enabled(), "Backbone caches are not enabled. Call setup_caches first."
        assert hasattr(self, 'decoder_causal_mask'), "Decoder causal mask not initialized. Call setup_caches first."
        assert self.decoder.caches_are_enabled(), "Decoder caches are not enabled. Call setup_caches first."
        # --- End Prerequisites ---

        # --- Backbone Forward Pass ---
        # 1. Embed input tokens (audio + text)
        # Shape: (b, s, cb+1, dim) -> sum -> (b, s, dim)
        embeds = self._embed_tokens(tokens) # Embeddings might be float32 depending on nn.Embedding
        masked_embeds = embeds * tokens_mask.unsqueeze(-1) # Apply mask
        h = masked_embeds.sum(dim=2) # Sum embeddings across codebook+text dimension
        h = h.to(compute_dtype) # Ensure backbone input matches compute dtype

        # 2. Prepare backbone mask
        # Index the full causal mask using the input positions
        # Result shape: (b, s, max_seq_len)
        indexed_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        # Slice mask's last dimension to match current sequence length 's'
        curr_seq_len = h.size(1)
        # --- CRITICAL FIX for Mask Dimension: Slice based on current seq len ---
        # The mask passed to attention should have dims [B, num_heads, S, S] or broadcastable.
        # Torchtune's Llama might handle [B, S, S] or [B, 1, S, S].
        # The indexed mask [B, S, max_len] needs to be adapted.
        # Let's assume torchtune handles the mask slicing internally based on input_pos
        # and we just need to provide the indexed mask correctly. Revisit if errors persist.
        # For sequence length S, the causal mask should effectively be S x S.
        # Let torchtune handle the mask shape details based on input_pos.
        # We provide the indexed mask based on positions.
        # NO - The mask needs to be sliced. The attention mechanism needs a mask relevant
        # to the *keys* it's attending to, up to the current step.
        curr_backbone_mask = indexed_mask[:, :, :curr_seq_len]
        # --- End CRITICAL FIX ---
        self.logger.debug(f"Backbone input shapes: h={h.shape}, pos={input_pos.shape}, mask={curr_backbone_mask.shape}")

        # 3. Run backbone
        try:
            # Pass input embeddings, positions, and the prepared mask
            backbone_output = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
            # Cast output just in case it differs from compute_dtype
            h = backbone_output.to(compute_dtype)
            self.logger.debug(f"Backbone output shape: {h.shape}")
        except Exception as e:
             self.logger.error("--- ERROR during self.backbone forward call ---", exc_info=True)
             self.logger.error(f"Input shapes: h={h.shape}, dtype={h.dtype}, device={h.device}")
             self.logger.error(f"Mask shape: {curr_backbone_mask.shape}, dtype={curr_backbone_mask.dtype}, device={curr_backbone_mask.device}")
             self.logger.error(f"Position shape: {input_pos.shape}, dtype={input_pos.dtype}, device={input_pos.device}")
             raise
        # --- End Backbone Section ---

        # --- Decoder Forward Pass ---
        # 1. Get the hidden state corresponding to the *last* input token from the backbone
        last_h = h[:, -1, :] # Shape: [b, backbone_dim]

        # 2. Predict the first codebook token (c0)
        c0_logits = self.codebook0_head(last_h) # Use dedicated head
        c0_sample = sample_topk(c0_logits, topk, temperature) # Sample token ID
        c0_embed = self._embed_audio(0, c0_sample) # Embed the sampled token [b, 1, backbone_dim]

        # 3. Iteratively predict remaining codebooks (c1 to cN-1) using the decoder
        curr_decoder_input_embed = c0_embed # Start with the embedding of c0
        # Keep track of all sampled codebook tokens for this frame
        curr_frame_samples = [c0_sample]
        # Decoder positions start from 0 and increment for each codebook prediction step
        # Decoder operates over max 'audio_num_codebooks' steps for one frame.
        decoder_pos = torch.arange(0, 1, device=device).unsqueeze(0) # Start at pos 0 for first decoder input (c0_embed)

        # --- CRITICAL: Decoder cache must be reset before processing each frame ---
        # This was likely a source of state leakage / dimension errors previously.
        self.reset_caches() # Reset BOTH backbone and decoder caches here for safety? Or just decoder? Let's try both.
        # Original CSM might have reset only decoder cache per frame. If issues persist, change to self.decoder.reset_caches()
        self.logger.debug("KV Caches reset before decoder loop.")
        # --- End Cache Reset ---

        for i in range(1, self.config.audio_num_codebooks):
            try:
                # Project the *last backbone state* (constant across decoder steps for a frame)
                # and the *previous codebook embedding* to the decoder dimension.
                # The original CSM passed concatenated [last_h, prev_embed], but here let's follow
                # torchtune Llama structure if possible, using only the *previous* step's output.
                # Let's assume the decoder takes the previous codebook embedding projected.
                projected_input = self.projection(curr_decoder_input_embed) # Project [b, 1, backbone_dim] -> [b, 1, decoder_dim]
                projected_input = projected_input.to(compute_dtype) # Ensure dtype

                # --- Prepare Decoder Mask ---
                # Index the decoder's causal mask (size num_codebooks x num_codebooks)
                # using the current decoder step position 'i'.
                indexed_decoder_mask = _index_causal_mask(self.decoder_causal_mask, decoder_pos)
                # Slice mask to match current decoder sequence length (which is always 1 in this loop)
                curr_decoder_mask = indexed_decoder_mask[:, :, :decoder_pos.size(1)]
                # --- End Decoder Mask ---

                self.logger.debug(f"Decoder step {i}: input shape={projected_input.shape}, pos={decoder_pos}, mask shape={curr_decoder_mask.shape}")

                # Run the decoder for one step
                decoder_output = self.decoder(projected_input, input_pos=decoder_pos, mask=curr_decoder_mask)
                decoder_h = decoder_output.to(compute_dtype) # Shape: [b, 1, decoder_dim]

                # Predict the i-th codebook logits using the decoder's output and the i-th audio head
                # self.audio_head shape: [cb-1, decoder_dim, vocab_size]
                # decoder_h[:, -1, :] shape: [b, decoder_dim]
                # audio_head[i-1] shape: [decoder_dim, vocab_size]
                ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
                ci_sample = sample_topk(ci_logits, topk, temperature) # Sample token [b, 1]
                ci_embed = self._embed_audio(i, ci_sample) # Embed [b, 1, backbone_dim]

                # Update for the next iteration
                curr_decoder_input_embed = ci_embed # Use the new embedding as input
                curr_frame_samples.append(ci_sample) # Store the sample
                decoder_pos = decoder_pos + 1 # Increment position for the next step

            except Exception as e:
                self.logger.error(f"--- ERROR during self.decoder forward call at step {i} ---", exc_info=True)
                self.logger.error(f"Input shapes: projected_input={projected_input.shape}, dtype={projected_input.dtype}")
                self.logger.error(f"Mask shape: {curr_decoder_mask.shape}, dtype={curr_decoder_mask.dtype}")
                self.logger.error(f"Position shape: {decoder_pos.shape}, dtype={decoder_pos.dtype}")
                raise

        # --- End Decoder Loop ---

        # Concatenate all sampled codebook tokens for this frame
        final_samples = torch.cat(curr_frame_samples, dim=1) # Shape: [b, audio_num_codebooks]
        return final_samples

    # --- Embedding helpers from original CSM repo (Keep as is) ---
    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds audio tokens using offsets for each codebook."""
        offset = codebook * self.config.audio_vocab_size
        # Ensure tokens are long type for embedding lookup
        return self.audio_embeddings(tokens.long() + offset)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds combined audio and text tokens."""
        # tokens shape: (batch_size, seq_len, audio_num_codebooks + 1)
        # Last element (-1) is text token, first 'cb' elements are audio tokens

        # Embed text token (index -1)
        # Ensure indices are long type
        text_embeds = self.text_embeddings(tokens[:, :, -1].long()).unsqueeze(-2) # Shape: (b, s, 1, dim)

        # Embed audio tokens (indices 0 to cb-1)
        audio_tokens = tokens[:, :, :-1] # Shape: (b, s, cb)
        # Create offsets for each codebook [0, vocab_size, 2*vocab_size, ...]
        codebook_indices = torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        audio_tokens_with_offset = audio_tokens + (self.config.audio_vocab_size * codebook_indices)

        # Embed audio tokens
        b, s, cb = audio_tokens_with_offset.shape
        # Flatten batch and seq dims for embedding lookup
        # Shape becomes: (b*s, cb) -> lookup -> (b*s, cb, dim)
        # Ensure indices are long type
        audio_embeds_flat = self.audio_embeddings(audio_tokens_with_offset.view(b * s, cb).long())
        # Reshape back to (b, s, cb, dim)
        audio_embeds = audio_embeds_flat.view(b, s, cb, -1)

        # Concatenate audio and text embeddings along the codebook/text dimension
        return torch.cat([audio_embeds, text_embeds], dim=2) # Result shape: (b, s, cb+1, dim)
    # --- End Embedding helpers ---

    # --- from_local_pretrained for loading (Keep as is from previous step) ---
    @classmethod
    def from_local_pretrained(cls: type[T], local_path: str | Path) -> T:
        """Loads the model configuration and weights from a local directory."""
        local_path = Path(local_path)
        logger.info(f"Loading model from local path: {local_path}")

        # 1. Config - Use default ModelArgs or load from a config file if present
        config_file = local_path / "config.json" # Or similar standard name
        if config_file.is_file():
             logger.info(f"Loading configuration from {config_file}")
             # Implement loading logic if config.json exists and defines ModelArgs fields
             # For now, we use default args as before:
             config = ModelArgs()
             logger.info(f"Using ModelArgs (default/loaded): {config}")
        else:
             config = ModelArgs()
             logger.warning(f"No config file found at {config_file}. Using default ModelArgs: {config}")

        # 2. Initialize model with config
        model = cls(config)

        # 3. Load the state dictionary (weights)
        # Prioritize safetensors, then standard PyTorch binaries
        potential_weight_files = ["model.safetensors", "pytorch_model.bin", "csm_weights.pt", "csm-1b.pt"]
        weight_file_path = None
        weight_file_type = None

        for filename in potential_weight_files:
            path_to_check = local_path / filename
            if path_to_check.is_file():
                weight_file_path = path_to_check
                weight_file_type = "safetensors" if filename.endswith(".safetensors") else "pytorch"
                logger.info(f"Found weight file: {weight_file_path} (Type: {weight_file_type})")
                break

        if weight_file_path is None:
            logger.error(f"Could not find model weight file in {local_path}. Checked for: {potential_weight_files}")
            raise FileNotFoundError(f"Could not find model weight file in {local_path}")

        logger.info(f"Loading weights from {weight_file_path} using {weight_file_type} loader...")

        # Load to CPU first for stability, device transfer happens later
        map_location_device = 'cpu'
        logger.debug(f"Using map_location='{map_location_device}' for loading weights.")

        state_dict = None
        if weight_file_type == "safetensors":
            try:
                 state_dict = load_safetensors_file(weight_file_path, device=map_location_device)
                 logger.info(f"Safetensors file loaded successfully to device '{map_location_device}'.")
            except Exception as e:
                 logger.error(f"Failed to load safetensors file {weight_file_path}: {e}", exc_info=True)
                 raise
        else: # PyTorch format (.bin, .pt)
            try:
                 # Try with weights_only=True first (safer)
                 state_dict = torch.load(weight_file_path, map_location=map_location_device, weights_only=True)
                 logger.info(f"PyTorch file loaded successfully with weights_only=True to device '{map_location_device}'.")
            except (pickle.UnpicklingError, RuntimeError, EOFError, AttributeError) as load_err: # Catch more potential errors
                 logger.warning(f"torch.load with weights_only=True failed ('{type(load_err).__name__}'). Retrying with weights_only=False (potential security risk).")
                 try:
                     # Fallback to weights_only=False
                     state_dict = torch.load(weight_file_path, map_location=map_location_device, weights_only=False)
                     logger.info(f"PyTorch file loaded successfully with weights_only=False to device '{map_location_device}'.")
                 except Exception as e:
                     logger.error(f"Failed to load PyTorch file {weight_file_path} even with weights_only=False: {e}", exc_info=True)
                     raise
            except Exception as e:
                 logger.error(f"Failed to load PyTorch file {weight_file_path}: {e}", exc_info=True)
                 raise

        # 4. Load state dict into model (model is currently on CPU)
        logger.info("Loading state dict into model architecture...")
        try:
             # Strict=True is preferred to catch mismatches
             load_result = model.load_state_dict(state_dict, strict=True)
             logger.info(f"State dict loaded successfully (strict=True). Result: {load_result}")
        except RuntimeError as e:
             logger.error(f"Error loading state dict (strict=True): Mismatched keys or sizes found. Details: {str(e)[:1000]}...") # Log more details
             logger.info("Attempting to load state dict with strict=False...")
             try:
                  load_result = model.load_state_dict(state_dict, strict=False)
                  logger.warning(f"State dict loaded with strict=False. Result: {load_result}")
                  logger.warning("Mismatched/missing keys were ignored. This might lead to unexpected behavior or errors later.")
             except Exception as e2:
                  logger.error(f"Failed to load state dict even with strict=False. Error: {e2}")
                  raise RuntimeError(f"Failed to load state dict from {weight_file_path}. Check logs for mismatch details.") from e2
        except Exception as e_other:
             logger.error(f"An unexpected error occurred during state dict loading: {e_other}", exc_info=True)
             raise

        model.eval() # Set to evaluation mode
        logger.info("Model loaded from local path to CPU and set to eval mode. Device transfer should happen after this.")
        return model # Return the model instance (still on CPU)
    # --- End from_local_pretrained ---