# ./scripts/test_models.py
import os
import json
import torch
import logging
from moshi.models.mimi import MimiModel  # Removed MimiConfig import
from moshi.models import loaders
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./scripts/test_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_file_exists(file_path, description):
    """Check if a file exists and log the result."""
    if os.path.exists(file_path):
        logger.info(f"{description} found at: {file_path}")
        return True
    else:
        logger.error(f"{description} NOT found at: {file_path}")
        return False

def load_and_log_mimi_model(mimi_config_path, device="cpu"):
    """Load the MimiModel and log its expected state dict keys."""
    if not check_file_exists(mimi_config_path, "Mimi config file"):
        return None

    # Load config
    try:
        with open(mimi_config_path, "r") as f:
            config_dict = json.load(f)
        logger.debug(f"Mimi config content: {json.dumps(config_dict, indent=2)}")
        # Initialize model with config dictionary
        model = MimiModel(config_dict)
        logger.info("Mimi config loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Mimi config: {e}")
        return None

    # Initialize model
    try:
        model.to(device)
        logger.info(f"MimiModel initialized on device: {device}")
    except Exception as e:
        logger.error(f"Failed to initialize MimiModel: {e}")
        return None

    # Log expected keys
    expected_keys = list(model.state_dict().keys())
    logger.info(f"Total expected keys: {len(expected_keys)}")
    logger.info("Expected state dictionary keys from MimiModel:")
    for key in expected_keys:
        logger.debug(f"  {key}")

    return model, expected_keys

def load_and_log_state_dict(mimi_weight_path):
    """Load the state dict from the weights file and log its keys."""
    if not check_file_exists(mimi_weight_path, "Mimi weights file"):
        return None

    try:
        pkg = torch.load(mimi_weight_path, map_location="cpu", weights_only=False)
        state_dict = pkg["model"]
        logger.info("State dictionary loaded from weights file")
    except Exception as e:
        logger.error(f"Failed to load state dict from {mimi_weight_path}: {e}")
        return None

    # Log provided keys
    provided_keys = list(state_dict.keys())
    logger.info(f"Total provided keys: {len(provided_keys)}")
    logger.info("Provided state dictionary keys from weights file:")
    for key in provided_keys:
        logger.debug(f"  {key}")

    return state_dict, provided_keys

def compare_keys(expected_keys, provided_keys):
    """Compare expected and provided keys and log differences."""
    missing_keys = [key for key in expected_keys if key not in provided_keys]
    unexpected_keys = [key for key in provided_keys if key not in expected_keys]

    logger.info(f"Total missing keys: {len(missing_keys)}")
    if missing_keys:
        logger.warning("Missing keys in state dict:")
        for key in missing_keys:
            logger.debug(f"  {key}")
    else:
        logger.info("No missing keys found in state dict")

    logger.info(f"Total unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        logger.warning("Unexpected keys in state dict:")
        for key in unexpected_keys:
            logger.debug(f"  {key}")
    else:
        logger.info("No unexpected keys found in state dict")

    return missing_keys, unexpected_keys

def main():
    logger.info("Starting model and weights verification script")

    base_dir = "./models"
    csm_dir = os.path.join(base_dir, "csm-1b")
    llama_dir = os.path.join(base_dir, "llama-3.2-1b")
    kyutai_dir = os.path.join(base_dir, "kyutai")
    csm_checkpoint = os.path.join(csm_dir, "ckpt.pt")
    llama_checkpoint = os.path.join(llama_dir, "ckpt.pt")
    mimi_weight_path = os.path.join(kyutai_dir, "mimi_weight.pt")
    mimi_config_path = os.path.join(kyutai_dir, "config.json")

    files_to_check = [
        (csm_dir, "CSM-1b model directory"),
        (csm_checkpoint, "CSM-1b checkpoint file"),
        (llama_dir, "Llama-3.2-1b model directory"),
        (llama_checkpoint, "Llama-3.2-1b checkpoint file"),
        (kyutai_dir, "Kyutai model directory"),
        (mimi_config_path, "Mimi config file"),
        (mimi_weight_path, "Mimi weights file"),
    ]

    all_files_present = True
    for path, desc in files_to_check:
        if not check_file_exists(path, desc):
            all_files_present = False

    if not all_files_present:
        logger.error("Some required files are missing. Downloading Mimi weights/config.")
        os.makedirs(kyutai_dir, exist_ok=True)
        try:
            hf_hub_download(repo_id="kyutai/mimi", filename="mimi_weight.pt", local_dir=kyutai_dir)
            hf_hub_download(repo_id="kyutai/mimi", filename="config.json", local_dir=kyutai_dir)
            logger.info("Downloaded mimi_weight.pt and config.json")
            check_file_exists(mimi_weight_path, "Mimi weights file (post-download)")
            check_file_exists(mimi_config_path, "Mimi config file (post-download)")
        except Exception as e:
            logger.error(f"Failed to download files: {e}")
            return

    model, expected_keys = load_and_log_mimi_model(mimi_config_path)
    if model is None:
        logger.error("Cannot proceed due to model loading failure")
        return

    state_dict, provided_keys = load_and_log_state_dict(mimi_weight_path)
    if state_dict is None:
        logger.error("Cannot proceed due to state dict loading failure")
        return

    missing_keys, unexpected_keys = compare_keys(expected_keys, provided_keys)

    try:
        model.load_state_dict(state_dict)
        logger.info("State dictionary loaded into MimiModel successfully")
    except RuntimeError as e:
        logger.error(f"Failed to load state dict into model: {e}")

    logger.info("Model and weights verification completed")

if __name__ == "__main__":
    main()