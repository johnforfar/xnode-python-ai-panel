import time
from datetime import datetime
import torchaudio
from env import data_dir
from tts import TTS
import os
import numpy as np
import asyncio

script = [
    {"speaker": "Kxi", "line": "Alright folks, let's pause the high-stakes debate for a quick 'Fun Segment'! We asked our panelists for their secret crypto confessions."},
    {"speaker": "Kai", "line": "I once almost bought a Bitcoin back in 2011. Decided gold was safer and I still think I was right."},
    {"speaker": "Liq", "line": "Oh, Kai, still polishing those pet rocks? I tried explaining Bitcoin to my cat. He just stared blankly, clearly he works for the fiat system."},
    {"speaker": "Vivi", "line": "I designed Bitcoin, but I still sometimes forget my private keys. Last week I had to reset my hardware wallet password."},
    {"speaker": "Nn", "line": "I told everyone I invented Bitcoin. Tremendous idea, everyone believed me, except the stupid people."},
    {"speaker": "Kxi", "line": "On to the next topic! If Bitcoin hit 1 million dollars tomorrow, what would you do?"},
    {"speaker": "Kai", "line": "Buy more gold, obviously! Maybe a slightly bigger vault."},
    {"speaker": "Liq", "line": "He'd secretly buy Bitcoin and pretend it was gold-plated! Don't try to deny it Kai."},
    {"speaker": "Vivi", "line": "Logically at $1 million volatility might decrease, making it a more stable asset layer. I would be very interested to see how the market will change."},
    {"speaker": "Nn", "line": "I'm not selling until it hits ten million. Folks, mark my words, I'll build a huge beautiful Bitcoin tower!"},
    {"speaker": "Kxi", "line": " And that concludes our very revealing 'Fun Segment'! Back to you Ashton!"},
]

speaker_id = {
    "Kxi": 0, # Moderator
    "Kai": 1, # Peter Schiff
    "Liq": 2, # Michael Saylor
    "Vivi": 3, # Satoshi Nakamoto
    "Nn": 4 # Donald Trump
}

async def play(broadcast_message):
    playAt = int(time.time()) + 10
    history = [{
        "agent": "System", "address": "system",
        "text": f"AI Panel: Crypto's Future.",
        "timestamp": datetime.utcnow().isoformat() + "Z", "audioStatus": "failed"
    }]
    await broadcast_message({"type": "conversation_history", "payload": {"history": history}})
    
    for i, message in enumerate(script):
        audio = f"/audio/replay/{i}.wav"
        history.append({
            "agent": message["speaker"],
            "address": "",
            "text": message["line"],
            "timestamp": datetime.fromtimestamp(playAt).isoformat() + "Z",
            "audioStatus": "ready",
            "audioUrl": audio,
        })

        
        await broadcast_message({"type": "conversation_history", "payload": {"history": history}})
        audio_data, sample_rate = torchaudio.load(f"{data_dir()}/static{audio}")
        chunk = audio_data.squeeze(0).numpy().astype(np.float32).tolist()
        await broadcast_message({"type": "audio", "payload": {"speaker": speaker_id[message["speaker"]], "playAt": playAt, "chunk": chunk}})
        playAt += (len(chunk) / 24000) + 0.5

async def broadcast_mock(message):
     return

async def generate():
    tts = TTS()
    for i, message in enumerate(script):
         file = await tts.generate_audio(message["line"], speaker_id[message["speaker"]], broadcast_mock, False)
         os.rename(file, f"{data_dir()}/static/audio/replay/{i}.wav")
         print(f"Generated replay message {i+1}/{len(script)}")

if __name__ == '__main__':
    asyncio.run(generate())