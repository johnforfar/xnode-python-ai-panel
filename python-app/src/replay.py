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
    {"speaker": "Kai", "line": "Confession? Fine. I once almost bought a Bitcoin... back in 2011. Decided gold was safer. (Scoffs) Still think I was right."},
    {"speaker": "Liq", "line": "Oh, Kai, still polishing those pet rocks? My confession: I tried explaining Bitcoin to my cat. He just stared blankly. Clearly, he works for the fiat system."},
    {"speaker": "Vivi", "line": "My turn. I designed Bitcoin... but I still sometimes forget my private keys. Had to reset my hardware wallet password last week. Embarrassing."},
    {"speaker": "Nn", "line": "Forget keys? Sad! My confession? I told everyone I invented Bitcoin. Tremendous idea, folks! Everyone believed me. Mostly."},
    {"speaker": "Kxi", "line": "Nn, claiming inventions again? Shocking! Okay, Kai, quick - if Bitcoin hit $1 million tomorrow, what would you really do?"},
    {"speaker": "Kai", "line": "...Buy more gold. Obviously. Maybe a slightly bigger vault."},
    {"speaker": "Liq", "line": "(Laughing) He'd secretly buy Bitcoin and pretend it was gold-plated! Classic Kai."},
    {"speaker": "Vivi", "line": "Logically, at $1 million, volatility might decrease, making it a more stable asset layer. Interesting data point."},
    {"speaker": "Nn", "line": "One million? We're going to ten million, folks! Believe me! I'd build a huge, beautiful Bitcoin tower! The best tower!"},
    {"speaker": "Kxi", "line": "A Bitcoin tower... okay Nn. And that concludes our very revealing 'Fun Segment'! Back to the serious debate... or maybe not!"},
]

speaker_id = {"Kxi": 0, "Kai": 1, "Liq": 2, "Vivi": 3, "Nn": 4}

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
        playAt += len(chunk) * 0.08 + 0.5

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