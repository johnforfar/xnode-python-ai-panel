import time
from datetime import datetime
import torchaudio
from env import data_dir
from tts import TTS
import os
import numpy as np
import asyncio

script = [
    {"order": "01", "speaker": "CryptoKitty", "line": "Alright, panel, welcome to the AI Panel. First question: what’s the messiest crypto confession you’re willing to admit? No PR spin—hit us with it."},
    {"order": "02", "speaker": "PeterGoldBug", "line": "Back in 2011, I almost bought Bitcoin. Nearly fell for the hype like a damn fool. Then I snapped out of it, stuck with gold, the real deal, while you crypto freaks jerk off to crashing charts."},
    {"order": "03", "speaker": "MrLightning", "line": "Peter, you fossil, still humping that gold bar? I sold my soul to Bitcoin—maxed every line of credit. Darkest secret? I’d do it again, even if it tanks, while you polish relics and miss the future."},
    {"order": "04", "speaker": "PeterGoldBug", "line": "You sanctimonious prick, Bitcoin’s a scam—tulips on crack. I’d rather bury gold in my backyard than touch your digital monopoly money. You’re no visionary, just a bagholder with a podcast."},
    {"order": "05", "speaker": "MrLightning", "line": "Adorable, Peter, like a caveman yelling at fire. Bitcoin’s eating your lunch, and I laugh at your tweets while stacking sats. Cry harder."},
    {"order": "06", "speaker": "RealSatoshi", "line": "I built Bitcoin to screw over banks, but here’s the kicker: lost half my stash to a dead hard drive in 2010. Genius move, right? Still beats trusting some suit."},
    {"order": "07", "speaker": "TheDon", "line": "Bitcoin? I made it happen—best idea ever, folks. Secret? I told everyone it’s huge, massive potential, and they ate it up. Don’t own any—don’t need to. I’m the brand, baby."},
    {"order": "08", "speaker": "CryptoKitty", "line": "Alright, you animals, next question: if Bitcoin hits a million bucks tomorrow, what’s your move? Go."},
    {"order": "09", "speaker": "PeterGoldBug", "line": "I’d buy gold, obviously—maybe a bigger safe to watch you idiots crash when the bubble pops. Bitcoin at a million? Still a scam, just with fancier zeros."},
    {"order": "10", "speaker": "MrLightning", "line": "Peter, you’d secretly buy Bitcoin and brag about it—don’t front. I’d leverage every dime, build a Bitcoin empire, and laugh as your gold turns into doorstops. Million’s just the start, peasant."},
    {"order": "11", "speaker": "RealSatoshi", "line": "I’d cash out a chunk—quietly. Fix some mistakes, maybe fund something new. Bitcoin’s not about me anymore; it’s yours to screw up."},
    {"order": "12", "speaker": "TheDon", "line": "I’d make it the official U.S. currency—tremendous, the best. Everyone’s saying, ‘Sir, you’re a genius,’ and they’re right. Gold? Old news. Bitcoin’s the new money now."},
    {"order": "13", "speaker": "CryptoKitty", "line": "And that’s where we cut it. AI Panel’s over—thanks for the chaos. Back to you, Ashton."},
]


speaker_id = {
    "CryptoKitty": 0,    # Moderator (previously named Kxi)
    "MrLightning": 1,    # Michael Saylor (previously named Liq)
    "PeterGoldBug": 2,   # Peter Schiff (previously named Kai)
    "RealSatoshi": 3,    # Satoshi Nakamoto (previously named Vivi)
    "TheDon": 4          # Donald Trump (previously named Nn)
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