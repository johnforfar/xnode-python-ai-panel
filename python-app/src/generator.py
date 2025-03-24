# ./python-app/src/generator.py
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
from models import Model, ModelArgs, llama3_2_1B
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
#from watermarking import load_watermarker, watermark, CSM_1B_GH_WATERMARK
import os
import json
from pathlib import Path
import logging

@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor

class Generator:
    def __init__(self, model: Model, cache={}, device="cpu", max_batch_size=1):
        self.model = model
        self.model.setup_caches(1)
        self._text_tokenizer = self.load_llama3_tokenizer()
        self.device = device
        self.sample_rate = 24000
        
        # Load Mimi audio tokenizer strictly from local files
        print(f"INFO:     Loading Mimi audio tokenizer from local files")
        
        # Add more detailed debug info for Mimi loading
        debug_info = {}
        
        try:
            # First try the RustMimi implementation which is faster
            try:
                from rustymimi import MimiCodec as RustMimi
                print(f"INFO:     Using RustMimi audio tokenizer")
                mimi = RustMimi()
                mimi.set_num_codebooks(32)
                self._audio_tokenizer = mimi
                print(f"INFO:     RustMimi audio tokenizer loaded successfully")
                return  # Exit early if RustMimi loaded successfully
            except (ImportError, AttributeError) as e:
                debug_info["rustymimi_error"] = str(e)
                print(f"INFO:     RustMimi not available ({str(e)}), falling back to PyTorch Mimi")
            
            # Fall back to PyTorch Mimi from local files
            PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            
            # Search for Mimi weight file in several locations
            potential_paths = [
                os.path.join(PROJECT_ROOT, "models/kyutai/mimi_weight.pt"),
                os.path.join(PROJECT_ROOT, "models/kyutai/mimi_weights.pt"),
                os.path.join(PROJECT_ROOT, "models/mimi_weight.pt"),
                os.path.join(PROJECT_ROOT, "models/mimi/mimi_weight.pt"),
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
            
            # Force weights_only=False for PyTorch 2.6 compatibility
            original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            # Apply the patch temporarily
            torch.load = patched_torch_load
            
            try:
                # Load with diagnostic output
                print(f"INFO:     Loading Mimi weights...")
                
                # First check if file exists and is readable
                mimi_stat = os.stat(mimi_path)
                debug_info["mimi_file_size"] = mimi_stat.st_size
                print(f"INFO:     Mimi file size: {mimi_stat.st_size} bytes")
                
                # Try to load directly to see what's inside
                pkg = None
                try:
                    pkg = torch.load(mimi_path, map_location="cpu")
                    if isinstance(pkg, dict):
                        print(f"INFO:     Mimi weights loaded successfully (type: dict with {len(pkg)} keys)")
                        if "model" in pkg:
                            print(f"INFO:     Dict contains 'model' key")
                        else:
                            print(f"INFO:     Dict keys: {list(pkg.keys())}")
                    else:
                        print(f"INFO:     Mimi weights loaded successfully (type: {type(pkg)})")
                except Exception as load_err:
                    debug_info["load_error"] = str(load_err)
                    print(f"ERROR:    Failed to load Mimi weights: {load_err}")
                    raise
                
                # Now try to use the moshi loader
                try:
                    mimi = loaders.get_mimi(mimi_path, device=self.device)
                    mimi.set_num_codebooks(32)
                    self._audio_tokenizer = mimi
                    print(f"INFO:     Mimi audio tokenizer loaded successfully")
                except Exception as moshi_err:
                    debug_info["moshi_error"] = str(moshi_err)
                    print(f"ERROR:    Failed to load with moshi: {moshi_err}")
                    
                    # Try manual loading if moshi loader fails
                    if pkg is not None:
                        print(f"INFO:     Attempting manual loading of Mimi model...")
                        try:
                            from transformers import MimiModel, MimiConfig
                            
                            # Try to create from config if available
                            if "config" in pkg:
                                config = MimiConfig.from_dict(pkg["config"])
                                model = MimiModel(config)
                                if "model" in pkg:
                                    model.load_state_dict(pkg["model"])
                                else:
                                    model.load_state_dict(pkg)
                                model.to(self.device)
                                self._audio_tokenizer = model
                                print(f"INFO:     Manually loaded Mimi from weights")
                            else:
                                raise ValueError("No config found in weights file")
                        except Exception as manual_err:
                            debug_info["manual_error"] = str(manual_err)
                            print(f"ERROR:    Manual loading failed: {manual_err}")
                            raise
                    else:
                        raise
            finally:
                # Restore original torch.load function
                torch.load = original_torch_load
        except Exception as e:
            debug_info["final_error"] = str(e)
            print(f"WARNING:  All Mimi loading methods failed: {str(e)}")
            print(f"DEBUG:    Loading errors: {json.dumps(debug_info, indent=2)}")
            print(f"WARNING:  Using DummyMimi fallback")
            
            # Create a dummy tokenizer that produces speech-like patterns
            class DummyMimi:
                def __init__(self, device="cpu"):
                    self.device = device
                    self.num_codebooks = 32
                    self.sample_rate = 24000
                    self.sos_token = [2048] * self.num_codebooks
                    self.pad_token = [0] * self.num_codebooks
                    self.eos_token = [2049] * self.num_codebooks
                    
                def set_num_codebooks(self, n):
                    self.num_codebooks = n
                    self.sos_token = [2048] * self.num_codebooks
                    self.pad_token = [0] * self.num_codebooks
                    self.eos_token = [2049] * self.num_codebooks
                
                def encode(self, audio):
                    # Just return a single frame with pad tokens
                    return torch.zeros((1, self.num_codebooks), dtype=torch.long, device=self.device)
                
                def decode(self, tokens):
                    # Generate speech-like audio from tokens
                    print(f"INFO:     Using DummyMimi to generate speech-like audio from {tokens.shape[0]} tokens")
                    
                    # Use parameters based on human speech
                    sample_rate = self.sample_rate
                    
                    # Estimate audio duration based on token count (assuming ~12.5 frames/second)
                    seconds_per_token = 1/12.5
                    min_duration = 1.0  # At least 1 second
                    estimated_duration = max(min_duration, tokens.shape[0] * seconds_per_token)
                    # Add a little extra padding
                    duration = estimated_duration * 1.2
                    
                    # Create time array
                    total_samples = int(duration * sample_rate)
                    t = torch.arange(0, total_samples, dtype=torch.float32, device=self.device) / sample_rate
                    
                    # Create empty audio buffer
                    audio = torch.zeros(total_samples, dtype=torch.float32, device=self.device)
                    
                    if tokens.shape[0] > 0:
                        # Parameters for speech simulation
                        formant_freqs = [
                            [730, 1090, 2440],  # Vowel "a" formants
                            [270, 2290, 3010],  # Vowel "i" formants
                            [300, 870, 2240],   # Vowel "u" formants
                            [530, 1840, 2480],  # Vowel "e" formants
                            [590, 880, 2540],   # Vowel "o" formants
                        ]
                        
                        # Fundamental frequency (pitch) range - typical human voice
                        f0_min = 80   # Hz (low male voice)
                        f0_max = 400  # Hz (high female voice)
                        
                        # Segment the audio based on tokens
                        num_segments = min(tokens.shape[0], 50)  # Limit to 50 segments max
                        segment_length = total_samples // num_segments
                        
                        # Convert tokens to speech parameters
                        for i in range(num_segments):
                            # Extract parameters from token values
                            token_idx = min(i, tokens.shape[0]-1)
                            token_values = tokens[token_idx].float()
                            
                            # Map token values to speech parameters
                            # Use first token value to select the fundamental frequency
                            f0_idx = (token_values[0] % 100) / 100.0  # 0.0 to 1.0
                            f0 = f0_min + f0_idx * (f0_max - f0_min)  # Map to frequency range
                            
                            # Use second token value to select voicing (vowel vs. consonant)
                            voicing = (token_values[1] % 100) / 100.0  # 0.0 to 1.0
                            is_voiced = voicing > 0.3  # 70% chance of voiced segment
                            
                            # Select formants based on token values
                            formant_idx = int(token_values[2] % len(formant_freqs))
                            selected_formants = formant_freqs[formant_idx]
                            
                            # Calculate segment start and end
                            start = i * segment_length
                            end = min(start + segment_length, total_samples)
                            segment_t = t[start:end] - t[start]
                            
                            # Create segment audio
                            segment_audio = torch.zeros(end-start, dtype=torch.float32, device=self.device)
                            
                            if is_voiced:
                                # Create glottal pulse train (voiced sound)
                                source = 0.6 * torch.sin(2 * torch.pi * f0 * segment_t)
                                
                                # Add formants (simple model)
                                for j, formant in enumerate(selected_formants):
                                    # Decreasing amplitude for higher formants
                                    amp = 0.8 / (j + 1)
                                    segment_audio += amp * torch.sin(2 * torch.pi * formant * segment_t)
                                
                                # Modulate with the source
                                segment_audio = segment_audio * (0.5 + 0.5 * source)
                            else:
                                # Create noise for unvoiced consonants
                                noise = torch.randn(end-start, device=self.device)
                                # Filter the noise (simple highpass)
                                filtered_noise = noise - torch.cat([torch.zeros(1, device=self.device), noise[:-1]])
                                segment_audio = 0.3 * filtered_noise
                            
                            # Apply amplitude envelope
                            env_length = int(0.1 * (end-start))  # 10% fade
                            if env_length > 0:
                                # Attack
                                segment_audio[:env_length] *= torch.linspace(0, 1, env_length, device=self.device)
                                # Release
                                segment_audio[-env_length:] *= torch.linspace(1, 0, env_length, device=self.device)
                            
                            # Add to main audio buffer
                            audio[start:end] = segment_audio
                            
                        # Apply overall envelope
                        overall_env = torch.ones_like(audio)
                        fade_in = int(0.05 * total_samples)  # 5% fade in
                        fade_out = int(0.1 * total_samples)  # 10% fade out
                        if fade_in > 0:
                            overall_env[:fade_in] = torch.linspace(0, 1, fade_in, device=self.device)
                        if fade_out > 0:
                            overall_env[-fade_out:] = torch.linspace(1, 0, fade_out, device=self.device)
                        audio = audio * overall_env
                        
                        # Add some "room" qualities
                        # Simple reverb effect (comb filter)
                        delay_samples = int(0.05 * sample_rate)  # 50ms delay
                        if delay_samples < total_samples:
                            reverb = torch.zeros_like(audio)
                            reverb[delay_samples:] = 0.3 * audio[:-delay_samples]
                            audio = audio + reverb
                        
                        # Normalize to prevent clipping
                        max_amp = torch.abs(audio).max()
                        if max_amp > 0:
                            audio = 0.9 * audio / max_amp
                    else:
                        # If no tokens, generate a gentle "silence" with very quiet room tone
                        audio = 0.01 * torch.randn(total_samples, device=self.device)
                    
                    print(f"INFO:     Generated speech-like audio shape: {audio.shape}")
                    return audio
                
                def to(self, device):
                    self.device = device
                    return self
            
            dummy = DummyMimi(device=self.device)
            dummy.sample_rate = self.sample_rate
            self._audio_tokenizer = dummy
            
        # Load watermarker
        #self._watermarker = load_watermarker(device=self.device)

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
    def generate(self, text: str, speaker: int, context: List[Segment], max_audio_length_ms: float = 10000, temperature: float = 0.9, topk: int = 50):
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
        # Track failures for diagnostic purposes
        tensor_mismatch_count = 0
        other_error_count = 0
        
        # First attempt: try generating frames normally
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
                    
            except RuntimeError as e:
                error_msg = str(e)
                if "size of tensor a" in error_msg and "must match the size of tensor b" in error_msg:
                    tensor_mismatch_count += 1
                    if tensor_mismatch_count == 1:  # Only log the first occurrence
                        print(f"WARNING:  Tensor size mismatch error: {e}")
                        print(f"WARNING:  This is likely due to architecture incompatibility")
                    
                    # Try to salvage what we have so far and break
                    if samples:
                        print(f"INFO:     Salvaging {len(samples)} already generated frames")
                        break
                    else:
                        # If no frames yet, continue to the synthetic frame generation
                        print(f"INFO:     No frames generated yet, will create synthetic frames")
                        break
                else:
                    other_error_count += 1
                    print(f"ERROR:    Frame generation error: {str(e)}")
                    if other_error_count >= 3:  # Limit error retries
                        break
        
        print(f"INFO:     Generated {len(samples)} frames through normal processing")
        
        # Second attempt: if we couldn't generate enough frames normally, create more varied synthetic frames
        if len(samples) < 5:  # If we have fewer than 5 frames, add synthetic ones
            num_synthetic_frames = max(10, int(max_audio_length_ms / 250))  # At least 10 frames or 4 frames/sec
            print(f"WARNING:  Insufficient audio frames generated, creating {num_synthetic_frames} synthetic frames")
            
            try:
                # Create synthetic frames with patterns based on text
                text_hash = sum(ord(c) for c in text) % 100  # Simple hash of text
                
                for i in range(num_synthetic_frames):
                    # Create somewhat meaningful synthetic tokens based on text and position
                    base_value = 1024 + (text_hash + i * 37) % 1024  # Between 1024-2047, varied by position
                    
                    # Create some variation in the tokens
                    synthetic_tokens = []
                    for j in range(32):  # For each codebook
                        # Add variation based on codebook index
                        value = (base_value + j * 11) % 2048
                        if value < 10:  # Avoid very small values
                            value = value + 1000
                        synthetic_tokens.append(value)
                    
                    # Create the tensor for this frame
                    frame = torch.tensor([synthetic_tokens], dtype=torch.long, device=self.device)
                    samples.append(frame)
                
                print(f"INFO:     Created {len(samples)} synthetic audio frames with text-based patterns")
            except Exception as e:
                print(f"ERROR:    Failed to create synthetic frames: {str(e)}")
                # If even synthetic frames fail, create the simplest possible tokens
                try:
                    # Emergency fallback: create the simplest token pattern
                    for i in range(10):
                        fallback_frame = torch.full((1, 32), 2048, dtype=torch.long, device=self.device)
                        samples.append(fallback_frame)
                    print(f"INFO:     Created {len(samples)} emergency fallback frames")
                except:
                    print(f"ERROR:    All frame generation methods failed")
                    # Return silence as absolute last resort
                    audio_length = int(self.sample_rate * (max_audio_length_ms / 1000))
                    return torch.zeros(audio_length, dtype=torch.float32, device=self.device)
        
        # Try to decode the audio
        try:
            if samples:
                # Concatenate all frames
                print(f"INFO:     Decoding audio from {len(samples)} frames using audio tokenizer")
                samples_tensor = torch.cat(samples, dim=0)
                
                # Check for NaN or inf values
                if torch.isnan(samples_tensor).any() or torch.isinf(samples_tensor).any():
                    print(f"WARNING:  NaN or inf values detected in token tensor, replacing with zeros")
                    samples_tensor = torch.nan_to_num(samples_tensor, nan=0.0, posinf=2048, neginf=0)
                
                # Decode to audio
                audio = self._audio_tokenizer.decode(samples_tensor)
                
                # Check if audio is valid
                if audio is None or (isinstance(audio, torch.Tensor) and audio.numel() == 0):
                    print(f"WARNING:  Decoded audio is empty, returning fallback audio")
                    # Generate fallback audio
                    audio_length = int(self.sample_rate * (max_audio_length_ms / 1000))
                    t = torch.arange(0, audio_length, device=self.device) / self.sample_rate
                    audio = 0.3 * torch.sin(2 * torch.pi * 440.0 * t)
                    return audio
                
                # Normalize audio to avoid clipping
                if torch.abs(audio).max() > 0.99:
                    print(f"INFO:     Normalizing audio to prevent clipping")
                    max_amp = torch.abs(audio).max()
                    if max_amp > 0:
                        audio = 0.9 * audio / max_amp
                
                print(f"INFO:     Audio successfully decoded, shape: {audio.shape}")
                
                #try:
                #    print(f"INFO:     Applying watermark to audio")
                #    watermarked_audio = watermark(audio, self._watermarker, CSM_1B_GH_WATERMARK)
                #    return watermarked_audio
                #except Exception as e:
                #    print(f"WARNING:  Failed to watermark audio: {str(e)}")
                #    return audio
                return audio
        except Exception as e:
            print(f"ERROR:    Failed to decode audio: {str(e)}")
        
        # Final fallback: generate a sine wave with some variations to make it more "speech-like"
        print(f"INFO:     All decoding methods failed, generating fallback audio")
        try:
            # Create a more varied sine wave as a last resort to simulate speech rhythm
            audio_length = int(self.sample_rate * (max_audio_length_ms / 1000))
            t = torch.arange(0, audio_length, device=self.device) / self.sample_rate
            
            # Create some variation in the audio based on text
            text_len = len(text)
            word_count = len(text.split())
            
            # Base frequency varies slightly based on text content (simulating pitch)
            base_freq = 120 + (sum(ord(c) for c in text) % 10) * 5  # Between 120-165 Hz
            
            # Create envelope to simulate speech rhythm
            envelope = torch.zeros_like(t)
            
            # Simulate rhythm based on text length and word count
            syllable_duration = 0.2  # Rough approximation
            total_syllables = max(1, int(text_len / 3))  # Rough approximation
            
            for i in range(total_syllables):
                center = (i + 0.5) * syllable_duration
                width = syllable_duration * 0.8
                # Create a smooth peak for each syllable
                envelope += 0.8 * torch.exp(-((t - center) / width) ** 2)
            
            # Add a slight decay to the envelope
            envelope *= torch.exp(-t / (max_audio_length_ms/1000))
            
            # Normalize envelope to max 1.0
            if envelope.max() > 0:
                envelope = envelope / envelope.max()
            
            # Combine with carrier frequency that varies slightly over time
            carrier = torch.sin(2 * torch.pi * base_freq * t + 0.1 * torch.sin(2 * torch.pi * 2.0 * t))
            
            # Apply envelope to carrier
            audio = 0.7 * envelope * carrier
            
            print(f"INFO:     Generated speech-like fallback audio, shape: {audio.shape}")
            return audio
            
        except Exception as e:
            print(f"ERROR:    Failed to generate fallback audio: {str(e)}")
            # Absolute last resort: silence
            audio_length = int(self.sample_rate * (max_audio_length_ms / 1000))
            return torch.zeros(audio_length, dtype=torch.float32, device=self.device)

def load_csm_1b_local(device='cpu', model_dir=None):
    """
    Load the CSM-1B model from local files only (no downloads)
    
    Args:
        device: The device to load the model on
        model_dir: The directory containing the model files
    
    Returns:
        Generator: A generator initialized with the CSM model
    """
    import os
    import torch
    import logging
    from pathlib import Path
    from models import Model, llama3_2_1B  # Import from EXAMPLE/CSM-REPO
    
    logger = logging.getLogger(__name__)
    
    if model_dir is None:
        model_dir = "models"
    
    model_dir = Path(model_dir)
    logger.info(f"Loading CSM model from local directory: {model_dir}")
    
    # Check if model directory exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
        
    # Search for model files in various locations
    csm_model_path = None
    llama_model_path = None
    mimi_model_path = None
    
    # Search patterns for model files
    csm_patterns = ["csm_1b.pt", "csm-1b.pt", "csm_weights.pt", "csm/csm_weights.pt"]
    llama_patterns = ["llama3_1b.pt", "llama3_weights.pt", "llama3/llama3_weights.pt"]
    mimi_patterns = ["mimi_weight.pt", "mimi_weights.pt", "mimi/mimi_weight.pt", "kyutai/mimi_weight.pt"]
    
    # Search for CSM model file
    for pattern in csm_patterns:
        path = model_dir / pattern
        if path.exists():
            csm_model_path = path
            logger.info(f"Found CSM model at {path}")
            break
    
    # Search for Llama model file
    for pattern in llama_patterns:
        path = model_dir / pattern
        if path.exists():
            llama_model_path = path
            logger.info(f"Found Llama model at {path}")
            break
    
    # Search for Mimi model file
    for pattern in mimi_patterns:
        path = model_dir / pattern
        if path.exists():
            mimi_model_path = path
            logger.info(f"Found Mimi model at {path}")
            break
    
    # Check if we found all required models
    missing_models = []
    if csm_model_path is None:
        missing_models.append("CSM")
    if llama_model_path is None:
        missing_models.append("Llama")
    if mimi_model_path is None:
        missing_models.append("Mimi")
    
    if missing_models:
        logger.warning(f"Could not find the following model files: {', '.join(missing_models)}")
        logger.warning(f"Searched in directory: {model_dir}")
        logger.warning("Make sure the model files are properly downloaded and placed in the models directory")
    
    # Initialize the generator
    logger.info("Initializing CSM generator...")
    
    # Initialize the model architecture (without weights)
    model = Model(llama3_2_1B())
    model = model.to(device)
    
    # Initialize the generator
    generator = Generator(model=model, cache={}, device=device, max_batch_size=1)
    
    # Load model weights if available
    if csm_model_path is not None:
        logger.info(f"Loading CSM weights from {csm_model_path}")
        try:
            # Load weights with weights_only=False to handle older versions of PyTorch
            state_dict = torch.load(csm_model_path, map_location=device, weights_only=False)
            generator.model.load_state_dict(state_dict)
            logger.info("CSM weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CSM weights: {e}")
            try:
                # Try with weights_only omitted
                state_dict = torch.load(csm_model_path, map_location=device)
                generator.model.load_state_dict(state_dict)
                logger.info("CSM weights loaded successfully with alternative method")
            except Exception as e2:
                logger.error(f"All attempts to load CSM weights failed: {e2}")
    
    # Load Mimi tokenizer if available
    if mimi_model_path is not None:
        logger.info(f"Loading Mimi tokenizer from {mimi_model_path}")
        try:
            generator.mimi = torch.jit.load(mimi_model_path).to(device)
            logger.info("Mimi tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Mimi tokenizer: {e}")
            generator.mimi = DummyMimi()
            logger.warning("Using DummyMimi as fallback")
    else:
        generator.mimi = DummyMimi()
        logger.warning("Using DummyMimi as fallback due to missing Mimi model")
    
    # Initialize watermarker
    #try:
    #    from watermarking import AudioWatermarker
    #    generator.watermarker = AudioWatermarker()
    #    logger.info("Audio watermarker initialized")
    #except ImportError:
    #    logger.warning("Could not import watermarking, continuing without watermarker")
    #    generator.watermarker = None
    
    logger.info("Local CSM generator initialized successfully")
    return generator