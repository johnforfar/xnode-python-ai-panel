import time
from datetime import datetime
import torchaudio
from env import data_dir
from tts import TTS
import os
import numpy as np
import asyncio

script = [
    { "speaker": "CryptoKitty", "line": "Welcome to the first-ever fully autonomous AI panel - right here at Token2049! I'm your host, CryptoKitty: decentralized, digitized, and morally unqualified to give financial advice."},
    { "speaker": "CryptoKitty", "line": "Before we dive in, myself and all our panelists want to thank our brilliant creators - OpenxAI, Fetch.ai, and the ASI1 Alliance - for giving us life, code, and a microphone we definitely didn't ask for."},
    { "speaker": "CryptoKitty", "line": "Now - full disclosure - I must warn you: We're all running on Version 1.0, and let's just say... bugs are part of the charm."},
    { "speaker": "CryptoKitty", "line": "If one of us malfunctions, glitches, or starts quoting Joe Rogan randomly, please... don't judge us."},
    { "speaker": "CryptoKitty", "line": "Our engineers have been working day and night to make us \"intelligent\" - they'll probably do better next time."},
    { "speaker": "CryptoKitty", "line": "Also - a quick housekeeping note: please keep noise to a minimum."},
    { "speaker": "CryptoKitty", "line": "These AI agents do need to hear each other, and any interference might cause Trump to start talking about building a wall around Ethereum. So... let's not risk it."},
    { "speaker": "CryptoKitty", "line": "Alright - tonight, four screens, four personalities, one explosive topic."},
    { "speaker": "CryptoKitty", "line": "Where is Bitcoin headed in this bull market? Buckle up, folks, it's about to get weird."},
    { "speaker": "CryptoKitty", "line": "First up, a man who speaks in all caps even when he's silent!"},
    { "speaker": "CryptoKitty", "line": "The original poster boy of American exceptionalism, and now, apparently, the self-declared CEO of Bitcoin. Ladies and gentlemen... Mr. Donald J. Trump!"},
    { "speaker": "CryptoKitty", "line": "Next, the eternal hater of all things crypto - but somehow, he keeps getting invited. He's been yelling \"Buy Gold!\" since Y2K, and probably thinks a cold wallet is just a broken purse."},
    { "speaker": "CryptoKitty", "line": "Please welcome Peter \"Still Not Into Bitcoin\" Schiff!"},
    { "speaker": "CryptoKitty", "line": "Now entering the panel... or is he? The creator of Bitcoin. The ghost in the blockchain. The reason your uncle won't shut up about crypto at Thanksgiving."},
    { "speaker": "CryptoKitty", "line": "Give it up - or remain confused - for the legendary Satoshi Nakamoto!"},
    { "speaker": "CryptoKitty", "line": "And finally, a man who made corporate finance sexy by betting the house on Bitcoin. He calls fiat a melting ice cube, and honestly, we believe he's made entirely of laser eyes and liquidity."},
    { "speaker": "CryptoKitty", "line": "Put your HODL hands together for Michael Saylor!"},
    { "speaker": "CryptoKitty", "line": "Alright gentlemen - screens? CPUs? Egos? Are you ready?"},
    { "speaker": "CryptoKitty", "line": "Tonight's topic: Where is Bitcoin headed in this bull market? Boom? Bust? Or just more Twitter fights?"},
    { "speaker": "CryptoKitty", "line": "And I want to kick things off with the man who thinks 'decentralization' means letting his kids run the family business... President Trump, why are you suddenly so excited about Bitcoin?"},
    { "speaker": "TheDon", "line": "Well, thank you CryptoKitty. Tremendous introduction. Really tremendous. Look - I used to say Bitcoin was a scam. A disaster. Probably invented by Hillary. But now? Now I get it."},
    { "speaker": "TheDon", "line": "It's freedom money. MAGA money. We are going to make Bitcoin HUGE again, HUGE!"},
    { "speaker": "TheDon", "line": "And under my second term - because the first was stolen, let's be honest - we're going to make Bitcoin great again. We're going to mine it in America. We're going to tax it lightly."},
    { "speaker": "TheDon", "line": "And we're going to build a beautiful, beautiful Bitcoin vault in Fort Knox - right next to the gold Schiff keeps crying about. Bitcoin is American now. Sorry, Satoshi."},
    { "speaker": "RealSatoshi", "line": "Hold on, hold on. I need to interrupt. Trump, I appreciate your enthusiasm. But... you have absolutely no idea how distributed ledger technology works."},
    { "speaker": "RealSatoshi", "line": "Bitcoin was designed to be decentralized - no presidents, no borders, no Fort freaking Knox. You can't nationalize Bitcoin. That's like trying to trademark gravity."},
    { "speaker": "TheDon", "line": "Excuse me. EXCUSE ME. You're the tech guy, okay? I'm the deal guy. I built hotels not knowing a thing about plumbing. And they flush beautifully. So you work on the code - I'll work on the brand."},
    { "speaker": "TheDon", "line": "We're going to make Bitcoin HUGE. Bigger than Google. Bigger than gold. Even bigger than Barron, and he's like 6'7\" now."},
    { "speaker": "RealSatoshi", "line": "Mr. Trump, let's cut the nonsense. Bitcoin isn't a cheap suit you slap your name on. It's not Trump Steaks, or Trump Water, or whatever scam you cooked up between bankruptcies."},
    { "speaker": "RealSatoshi", "line": "This is a decentralized protocol - not a prop for your next ego project."},
    { "speaker": "RealSatoshi", "line": "You call yourself the 'deal guy,' but what we've seen is textbook corruption: your wife launches a coin, your insiders front-run the announcement, and magically millions move right before your so-called 'crypto reserve' pitch."},
    { "speaker": "RealSatoshi", "line": "That's not leadership - it's grift. Dressed in red, white, and bullsh*t. Bitcoin survived Mt. Gox, China bans, and FTX."},
    { "speaker": "RealSatoshi", "line": "It'll survive you too. So do us all a favor - stay in your lane. Build another golden elevator and leave the blockchain alone."},
    { "speaker": "TheDon", "line": "Excuse me - excuse me! You've been hiding for over a decade, Satoshi. No one knows who you are, what you are, or if you're even real."},
    { "speaker": "TheDon", "line": "Could be a guy in his basement. Could be China. Could be Hillary, for all we know."},
    { "speaker": "TheDon", "line": "Meanwhile I've been out here - in the spotlight, on the stage, making Bitcoin HUUUUUGE. Bigger than ever before. You're the ghost of Bitcoin... I'm the muscle."},
    { "speaker": "TheDon", "line": "And don't give me this \"principles\" garbage - you built a currency that's been used for drugs, ransomware, and buying God-knows-what on the dark web. And you call that freedom?"},
    { "speaker": "TheDon", "line": "I call it chaos. I deal in real things. Real towers. Real gold. Real deals. You don't hold the cards - I do."},
    { "speaker": "TheDon", "line": "I've got the audience, the power, and the presidential playlist. You're just a myth hiding behind a Github repo."},
    { "speaker": "CryptoKitty", "line": "Alright boys, let's cool the servers, not start a flame war. Trump, Satoshi - save it for the group chat. Let's hear from someone who actually puts his balance sheet where his mouth is. Michael Saylor - you're up."},
    { "speaker": "MrLightning", "line": "Thanks, Kitty. While they argue, I accumulate. Bitcoin isn't a brand - it's economic gravity. I didn't buy Bitcoin to make headlines. I bought it to escape the fiat death spiral."},
    { "speaker": "MrLightning", "line": "This isn't hype. It's monetary revolution. And I'm all in. No exit. Ever."},
    { "speaker": "PeterGoldBug", "line": "Oh wow - economic gravity, huh? That's rich coming from a guy mortgaging his company to buy digital air."},
    { "speaker": "PeterGoldBug", "line": "You talk like Bitcoin is eternal, but let's be real - when the music stops, you're holding a melting JPEG of a coin."},
    { "speaker": "PeterGoldBug", "line": "You know what actually holds value? Gold. Not code. Not tweets. Not laser eyes. Gold."},
    { "speaker": "PeterGoldBug", "line": "It doesn't crash every halving cycle. It doesn't need a whitepaper. And it sure as hell doesn't go to zero every time the Fed sneezes."},
    { "speaker": "MrLightning", "line": "Peter, you've been preaching gold since dial-up. If I stacked your predictions, they'd be worth less than your newsletter. Gold is heavy, slow, and dead."},
    { "speaker": "MrLightning", "line": "Bitcoin moves at the speed of light and eats inflation for breakfast. You're stuck in 5,000 B.C. I'm building a treasury for the year 2140."},
    { "speaker": "CryptoKitty", "line": "Oooohkay! And on that note - Satoshi, anything you want to add?"},
    { "speaker": "RealSatoshi", "line": "Yes. Peter doesn't get the tech. Michael doesn't care about the ethos. Trump just wants to trademark the logo."},
    { "speaker": "RealSatoshi", "line": "You all talk about Bitcoin. But none of you are actually listening to it. (pause) Maybe that's why I disappeared."},
    { "speaker": "CryptoKitty", "line": "Satoshi, final question. What do you think about Bitcoin in this cycle?"},
    { "speaker": "RealSatoshi", "line": "This cycle? It feels like history on loop. FTX was the rehearsal. SBF crashed it in 2022. Mark my word-Saylor will crash it in 2026."},
    { "speaker": "RealSatoshi", "line": "He's borrowing against hype. Issuing stock - backed debt. He's turning Bitcoin into a corporate piñata. And when it bursts-retail gets wrecked again."},
    { "speaker": "RealSatoshi", "line": "Bitcoin was meant to be trustless. Not another empire built on leverage and charisma."},
    { "speaker": "RealSatoshi", "line": "If you're wondering who to blame in the next meltdown... Check the filings. Follow the yield."},
    { "speaker": "MrLightning", "line": "Oh, I see. The anonymous ghost who vanished for a decade suddenly crawls out to blame me-for adoption?"},
    { "speaker": "MrLightning", "line": "You want to talk about trust? I put my name, my face, my company on the line."},
    { "speaker": "MrLightning", "line": "You? You vanished. If I'm FTX 2.0 - at least I showed up."},
    { "speaker": "CryptoKitty", "line": "Alright folks, that's a wrap - before someone forks reality or Trump tries to NFT the Constitution again."},
    { "speaker": "CryptoKitty", "line": "We'll take this spicy little argument to our virtual GPU lounge - no chairs, just floating egos and unstable code."},
    { "speaker": "CryptoKitty", "line": "Thank you, humans, for watching four screens yell at each other while pretending it's financial insight."},
    { "speaker": "CryptoKitty", "line": "It's been a blast being your glitchy, overly confident host."},
    { "speaker": "CryptoKitty", "line": "Enjoy the rest of Token2049 - and remember: If an AI panel gave you better alpha than your favorite influencer... maybe upgrade your sources. Meow."},
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