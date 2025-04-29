import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from env import data_dir
import re
from generator import Segment, load_csm_1b
import asyncio
import numpy as np
import time

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
                SPEAKER_PROMPTS["conversational_a"]["text"],
                0,
                SPEAKER_PROMPTS["conversational_a"]["audio"],
                self.generator.sample_rate
            ),
            prepare_prompt(
                ("Remember George Washington? You know how he died? Well-meaning physicians bled him to death. "
                "And this was the most important patient in the country, maybe in the history of the country. "
                "And we bled him to death trying to help him. So when you're actually inflating the money supply at 7%, but you're calling it 2% "
                "because you want to help the economy, you're literally bleeding the free market to death."),
                1,
                f"{data_dir()}/voices/michael.wav",
                self.generator.sample_rate
            ),
           prepare_prompt(
                ("Well, they're not buying dollars. They're selling dollars. Gold is the safe haven. "
                "They're not buying treasuries either, because I've said this before, when inflation is the threat, there is no safety in treasuries. "
                "But the other problem, if the world now has moved away from dollars, away from treasuries, and away from Bitcoin, "
                "not that they ever embraced it, but I'll get back into that, gold is the only safe haven standing."),
                2,
                f"{data_dir()}/voices/peter.wav",
                self.generator.sample_rate
            ),
            prepare_prompt(
                SPEAKER_PROMPTS["conversational_b"]["text"],
                3,
                SPEAKER_PROMPTS["conversational_b"]["audio"],
                self.generator.sample_rate
            ),
            prepare_prompt(
                ("Then he said, about six months ago, he's better than Reckon, "
                "and then he said a few nights ago, he's the greatest we've ever had. "
                "I said, I said, does that include Lincoln and George Washington? "
                "He said, that includes them all. That's ludops."),
                4,
                f"{data_dir()}/voices/donald.wav",
                self.generator.sample_rate
            ),
        ]

        self.playAt = 0

    async def generate_audio(self, text, speaker_id, broadcast_message, usePlayAt = True):
        if usePlayAt:
            current_time = int(time.time()) 
            self.playAt = max(current_time + 10, self.playAt) # Set to current time (if lower than playAt)
            if current_time + 20 < self.playAt:
                # Generate content max 20 seconds in advance (not to overwhelm the client) 
                await asyncio.sleep(self.playAt - (current_time + 20))

        prompt_segments = self.prompt_segments.copy()
        mimic = speaker_id == 6
        if mimic:
            prompt_segments.append(prepare_prompt(
                ("Hello there, I am mimic. I will copy your voice and give myself several compliments. "
                "I bet you can't wait to see how the result turns out. "
                "I am not quite sure what text we should put here, anything less than 30 seconds should work. "
                "Make sure the content has a similar vibe all the way through that matches our output prompt."),
                6,
                f"{data_dir()}/voices/mimic.wav",
                24000
            ))

        audio_chunks = []
        for audio_chunk in self.generator.generate_stream(
            text=re.sub(r'[:;"*]| -', '', text), # Remove : ; " * -(with a space in front) characters as they mess up the speech
            speaker=speaker_id,
            # Only add this speakers prompt and last message to context
            context=next(([item] for item in prompt_segments if item.speaker == speaker_id), []) + next(([item] for item in reversed(self.generated_segments) if item.speaker == speaker_id), []),
            max_audio_length_ms=30_000,
        ):
            chunk = audio_chunk.cpu().numpy().astype(np.float32).tolist()
            await broadcast_message({"type": "audio", "payload": {"speaker": speaker_id, "playAt": self.playAt if usePlayAt else 0, "chunk": chunk}})
            audio_chunks.append(audio_chunk)

        if not mimic:
            audio_tensor = torch.cat(audio_chunks)
            self.generated_segments.append(Segment(text=text, speaker=speaker_id, audio=audio_tensor))

            output = f"{data_dir()}/static/audio/{len(self.generated_segments)}.wav"
            torchaudio.save(
                output,
                torch.cat([audio_tensor], dim=0).unsqueeze(0).cpu(),
                self.generator.sample_rate
            )
            if usePlayAt:
                self.playAt += len(audio_chunks) * 20 * 0.08 + 0.5 # Wait 0.08s per fragment (20 in one batch) + 0.5s between fragments
            return output