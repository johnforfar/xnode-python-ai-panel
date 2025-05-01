import os
import torch
import torchaudio
from env import data_dir
import re
from generator import Segment, load_csm_1b
import asyncio
import numpy as np
import time

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

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
                ("And here are the ingredients. Although olives don't go in it, the olives are for you to eat as you're making it. It's very simple. "
                "You'll need vodka, elderflower cordial, mint soda, some limes. And if you don't like vodka, you don't have to put it in. "
                "So we're gonna get cracking. You also need to have some form of a drink as you're getting the food ready for yourself. "
                "Anyway, and this is my personal favorite. That's not very helpful if you have a different glass, "
                "but give it a go and then I'm going to do you pick your leaves and then you're going to clap them."),
                0,
                f"{data_dir()}/voices/florence.wav",
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
                ("Corruption is everywhere, from patients using with staff members to staff members having affairs. "
                "Before I got hired, the guy that interviewed me appeared to be under the influence of stimulants. "
                "A lot of the patients that came in knew staff members because they were friends or maybe even they used together a long time ago. "
                "And unfortunately, a lot of those workers relapsed."),
                3,
                f"{data_dir()}/voices/satoshi.wav",
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

    async def generate_audio(self, text, speaker_id, broadcast_message, usePlayAt = True, extraContext = []):
        if usePlayAt:
            current_time = int(time.time()) 
            self.playAt = max(current_time + 10, self.playAt) # Set to current time (if lower than playAt)
            if current_time + 20 < self.playAt:
                # Generate content max 20 seconds in advance (not to overwhelm the client) 
                await asyncio.sleep(self.playAt - (current_time + 20))

        mimic = speaker_id > 5

        audio_chunks = []
        for audio_chunk in self.generator.generate_stream(
            text=re.sub(r'\.\.\.| - |; |: |---', ', ', re.sub(r'["*]', '', text)), # Remove/replace some characters as they mess up the speech
            speaker=speaker_id,
            # Only add this speakers prompt to context
            context=next(([item] for item in self.prompt_segments if item.speaker == speaker_id), []) + next(([item] for item in reversed(self.generated_segments) if item.speaker == speaker_id), []) + extraContext,
            max_audio_length_ms=30_000,
        ):
            chunk = audio_chunk.cpu().numpy().astype(np.float32).tolist()
            await broadcast_message({"type": "audio", "payload": {"speaker": speaker_id, "playAt": self.playAt if usePlayAt else 0, "chunk": chunk}})
            audio_chunks.append(audio_chunk)

        if not mimic:
            if len(audio_chunks) * 20 * 0.08 == 30_000:
                # Max duration, probably a bug, regenerate
                return self.generate_audio(text, speaker_id, broadcast_message, usePlayAt, extraContext)

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