import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from env import data_dir
import re

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

class TTS:
    def __init__(self):
        # Select the best available device, skipping MPS due to float64 limitations
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"Using device: {device}")

        # Load model
        self.generator = load_csm_1b(device)

        # Generate each utterance
        self.generated_segments = []
        self.prompt_segments = [
            prepare_prompt(
                SPEAKER_PROMPTS["conversational_b"]["text"],
                0,
                SPEAKER_PROMPTS["conversational_b"]["audio"],
                self.generator.sample_rate
            ),
            prepare_prompt(
                SPEAKER_PROMPTS["conversational_b"]["text"],
                1,
                SPEAKER_PROMPTS["conversational_b"]["audio"],
                self.generator.sample_rate
            ),
            prepare_prompt(
                SPEAKER_PROMPTS["conversational_b"]["text"],
                2,
                SPEAKER_PROMPTS["conversational_b"]["audio"],
                self.generator.sample_rate
            ),
            prepare_prompt(
                SPEAKER_PROMPTS["conversational_b"]["text"],
                3,
                SPEAKER_PROMPTS["conversational_b"]["audio"],
                self.generator.sample_rate
            ),
            prepare_prompt(
                SPEAKER_PROMPTS["conversational_b"]["text"],
                4,
                SPEAKER_PROMPTS["conversational_b"]["audio"],
                self.generator.sample_rate
            ),
        ]

    def generate_audio(self, text, speaker_id):
        print(f"Generating: {text}")
        audio_tensor = self.generator.generate(
            text=re.sub(r'[:;"]', '', text), # Remove : ; " characters as they mess up the speech
            speaker=speaker_id,
            # Only add this speakers prompt and last message to context
            context=next(([item] for item in self.prompt_segments if item.speaker == speaker_id), []) + next(([item] for item in reversed(self.generated_segments) if item.speaker == speaker_id), []),
            max_audio_length_ms=30_000,
        )
        self.generated_segments.append(Segment(text=text, speaker=speaker_id, audio=audio_tensor))

        output = f"{data_dir()}/static/audio/{len(self.generated_segments)}.wav"
        torchaudio.save(
            output,
            torch.cat([audio_tensor], dim=0).unsqueeze(0).cpu(),
            self.generator.sample_rate
        )
        return output