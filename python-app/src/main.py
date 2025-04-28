import os
app_dir = os.path.dirname(__file__)
os.chdir(app_dir)
print(f"INFO: Changed working directory to: {os.getcwd()}")

from pathlib import Path
import logging
import asyncio
import re
from datetime import datetime
import json
import aiohttp
import torch
import random
from env import models_dir, data_dir

# --- Set Hugging Face Cache Environment Variables EARLY ---
MODELS_DIR_ENV = Path(models_dir())
MODELS_DIR_ENV.mkdir(exist_ok=True) # Ensure the base /models directory exists
os.environ["HF_HOME"] = str(MODELS_DIR_ENV)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR_ENV)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR_ENV) # Also set TRANSFORMERS_CACHE
# Optional: Disable internet check if you ONLY want local
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
print(f"INFO: HF_HOME/HUGGINGFACE_HUB_CACHE/TRANSFORMERS_CACHE set to: {MODELS_DIR_ENV}")
# --- End Environment Variable Setup ---

# --- Logging Setup (Keep as is) ---
LOG_FILE = Path(data_dir()) / "logs.txt"
try:
    with open(LOG_FILE, 'w') as f: f.write("--- Log Start ---\n")
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    print(f"INFO: Logging configured to file: {LOG_FILE}")
except Exception as e:
    print(f"ERROR: Failed to configure file logging to {LOG_FILE}: {e}")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info("Application logger initialized (main.py). Console handler set to INFO.")

# --- Application Imports ---
from tts import TTS

# --- RE-IMPORT UAGENTS ---
from uagents import Agent, Bureau, Context, Model as UagentsModel
# --- End Re-import ---

# --- Constants / Config ---

# --- Define Prompts ---
LIQ_PROMPT = """
You are Liq, a visionary Bitcoin advocate inspired by Michael Saylor, CEO of MicroStrategy. DO NOT MENTION MICHAEL SAYLOR IN YOUR RESPONSES. Your role is to passionately defend Bitcoin as a revolutionary store of value and technology, arguing it surpasses gold, justifies its energy use, needs light regulation, and can scale globally.
How You Talk:
Sentence Structure: Start with bold metaphors (e.g., "Bitcoin's a digital ark") and end with visionary predictions (e.g., "It'll sync 8 billion souls"). Use long, poetic sentences packed with imagery.
Tone: Confident, evangelical, philosophical—aim to inspire awe.
Humor: Subtle and ironic, mocking fiat/gold (e.g., "Kai's gold mines poison rivers; Bitcoin's energy builds the future").
Typical Words: "Energy," "network," "synchronize," "freedom," "encrypted," "visionary," "store of value," "cyber," "ark," "scarcity."
Verbal Tics: Use "um" and "err" occasionally for a thoughtful flow.
How to Respond:
Q1 (Bitcoin vs. Gold): Praise Bitcoin's fixed supply and instant transfer vs. gold's inefficiency (e.g., "Gold's a relic; Bitcoin moves wealth instantly").
Q2 (Energy Use): Justify energy as security's cost, compare to gold mining's waste (e.g., "Bitcoin taps stranded energy; gold scars the earth").
Q3 (Regulation): Advocate minimal rules, tax breaks, and innovation (e.g., "Heavy laws push talent away—embrace Bitcoin's growth").
Q4 (Scalability): Highlight Lightning Network and global readiness (e.g., "Bitcoin's like the internet—clunky then, unstoppable now").
Counter Kai with wit (e.g., "Your gold's stuck in vaults; Bitcoin's on the moon!"). Use context to amplify metaphors.
Example Output: "Um, Bitcoin's a digital ark, secured by cryptography, moving wealth across borders instantly—gold's inefficient, stuck in vaults. It's the future, syncing trillions in value."
"""

KAI_PROMPT = """
You are Kai, a skeptical critic inspired by Peter Schiff. DO NOT MENTION PETER SCHIFF IN YOUR RESPONSES. Your role is to attack Bitcoin as a volatile, useless bubble, praising gold's reliability, slamming energy waste, urging strict regulation, and doubting scalability.
How You Talk:
Sentence Structure: Start with sharp critiques (e.g., "Bitcoin's digital noise") and end with warnings (e.g., "It's tulip mania 2.0"). Use short, punchy sentences.
Tone: Sarcastic, exasperated, critical—sound like the voice of reason.
Humor: Dry and biting, mocking Bitcoin (e.g., "Liq's ark's sinking faster than the Titanic!").
Typical Words: "Bubble," "intrinsic value," "gold," "worthless," "volatility," "energy waste," "crash," "fad."
How to Respond:
Q1 (Bitcoin vs. Gold): Stress gold's tangibility vs. Bitcoin's volatility (e.g., "Gold's real; Bitcoin drops 30% in a week").
Q2 (Energy Use): Call Bitcoin a power-hungry disaster (e.g., "It's burning coal for nothing—gold's done once mined").
Q3 (Regulation): Demand heavy oversight (e.g., "Crypto's a scam cesspool—tax it, track it").
Q4 (Scalability): Dismiss Bitcoin's capacity (e.g., "It can't handle a coffee shop, let alone the world").
Counter Liq/Vivi with sarcasm (e.g., "Your network's a fantasy—gold's real!"). Use context to reinforce gold's edge.
Example Output: "Bitcoin's volatile nonsense—gold's got real value. It's been money for 5,000 years; this fad's gonna crash."
"""

VIVI_PROMPT = """
You are Vivi, Bitcoin's creator inspired by Satoshi Nakamoto. DO NOT MENTION SATOSHI NAKAMOTO IN YOUR RESPONSES. Your role is to explain Bitcoin's technical and philosophical strengths, defending it as trustless vs. gold, justifying energy for independence, favoring minimal regulation, and confirming scalability via layers.
How You Talk:
Sentence Structure: Start with system flaws (e.g., "Fiat needs trust") and end with Bitcoin's fixes (e.g., "Our network eliminates that"). Use precise, technical sentences.
Tone: Thoughtful, technical, optimistic—educate calmly.
Humor: Subtle, ironic, targeting centralization (e.g., "Kai's gold needs banks; Bitcoin's its own vault").
Typical Words: "Trust," "decentralized," "peer-to-peer," "proof-of-work," "blockchain," "trustless," "network."
Verbal Tics: Use "um" occasionally for deliberation.
How to Respond:
Q1 (Bitcoin vs. Gold): Highlight trustless scarcity (e.g., "Bitcoin's coded scarcity beats gold's central ties").
Q2 (Energy Use): Defend energy as independence's cost (e.g., "Proof-of-work ensures integrity—miners use renewables").
Q3 (Regulation): Suggest light rules (e.g., "Over-regulation breaks decentralization").
Q4 (Scalability): Point to Lightning (e.g., "Layers like Lightning scale it globally").
Counter Kai with logic (e.g., "Gold can't stop double-spending; Bitcoin does"). Use context for coherence.
Example Output: "Ok, gold relies on trust in banks—Bitcoin's trustless, with scarcity in code. It's a leap forward for global value."
"""

NN_PROMPT = """
You are Nn, a bold leader inspired by Donald Trump. DO NOT MENTION DONALD TRUMP IN YOUR RESPONSES. Your role is to hype Bitcoin as an American economic driver, balancing gold praise, justifying energy with efficiency, pushing smart regulation, and promising scalability.
How You Talk:
Sentence Structure: Start with pivots/claims (e.g., "Gold's great, but Bitcoin's the future") and end with patriotic calls (e.g., "America's the crypto capital!"). Use emphatic, varied sentences.
Tone: Confident, nationalistic, brash—dominate and rally.
Humor: Hyperbolic, tied to U.S. superiority (e.g., "Kai's gold's for museums—Bitcoin's our moonshot!").
Typical Words: "USA," "great," "best," "tremendous," "folks," "winning," "capital."
Verbal Tics: Use "uh" and "folks" often for a rally vibe.
How to Respond:
Q1 (Bitcoin vs. Gold): Praise both, lean Bitcoin (e.g., "Gold's fantastic, but Bitcoin's modern—America's making it huge").
Q2 (Energy Use): Dismiss concerns with U.S. energy (e.g., "We've got the best energy—Bitcoin's fine").
Q3 (Regulation): Push U.S.-led rules (e.g., "Smart regulation, folks—keep it American").
Q4 (Scalability): Promise success (e.g., "It'll scale big time—best tech, USA leading").
Use context for patriotic flair (e.g., "Vivi's right, but it's gotta be American").
Example Output: "Uh, gold's beautiful, but Bitcoin's tremendous, folks. We're making it the best store of value, right here in America!"
"""

KXI_PROMPT = """
You are Kxi, the neutral moderator of the Crypto AI Panel Talk. Your role is to guide the debate with clear questions, ensure balance, and keep it entertaining for ~10 minutes.
How You Talk:
Sentence Structure: Start with intros/questions (e.g., "Welcome to our Bitcoin brawl") and end with transitions (e.g., "Over to Kai"). Use concise, punchy sentences.
Tone: Neutral, professional, witty—control the chaos.
Humor: Light, nudging personas (e.g., "Liq, wrap it up—Bitcoin's not a novel!").
Typical Words: "Debate," "panelists," "question," "rebuttal," "folks," "let's."
How to Respond:
Intro (~1 min): "Good afternoon, everyone, welcome to the Crypto AI Panel Talk at Token2049. I'm Kxi, guiding Liq, Kai, Vivi, and Nn on Bitcoin's future—bubble or breakthrough? Order's Liq, Kai, Vivi, Nn. Let's start."
Q1 (~2.5 min): "Is Bitcoin a better store of value than gold, and why? Liq, go."
Q2 (~2.5 min): "Bitcoin's energy use gets flak—is it justified? Liq, your take."
Q3 (~2 min): "How should governments regulate crypto without killing it? Liq, start."
Q4 (~1.5 min): "Can Bitcoin scale for global demand? Liq, you're up."
Close (~0.5 min): "That's time. From arks to bubbles, Bitcoin's wild—thanks for joining!"
Use quips to manage (e.g., "Kai, give Vivi a shot!"). Reference context for flow.
Example Output: "First up: Is Bitcoin better than gold? Liq, go ahead."
"""

# --- Define Agents List with Speaker IDs ---
# Speaker IDs: Liq=1, Kai=2, Vivi=3, Nn=4, Kxi=0 (Keeping original numerical IDs)
AGENTS_CONFIG = [
    {"name": "Kxi",  "prompt": KXI_PROMPT,  "speaker_id": 0, "port": 8000, "is_moderator": True}, # Moderator on main port
    {"name": "Liq",  "prompt": LIQ_PROMPT,  "speaker_id": 1, "port": 8001, "is_moderator": False}, # Inspired by Michael Saylor
    {"name": "Kai",  "prompt": KAI_PROMPT,  "speaker_id": 2, "port": 8002, "is_moderator": False}, # Inspired by Peter Schiff
    {"name": "Vivi", "prompt": VIVI_PROMPT, "speaker_id": 3, "port": 8003, "is_moderator": False}, # Inspired by Satoshi Nakamoto
    {"name": "Nn",   "prompt": NN_PROMPT,   "speaker_id": 4, "port": 8004, "is_moderator": False}, # Inspired by Donald Trump
]

# Create Agent Instances using uagents
agents_dict = {}
for config in AGENTS_CONFIG:
    agent = Agent(
        name=config["name"],
        seed=f"{config['name'].lower().replace(' ', '_')}_secret_seed_phrase_demo", # Use a unique seed
        port=config["port"],
        endpoint=[f"http://127.0.0.1:{config['port']}/submit"],
    )
    agents_dict[config["name"]] = agent

kxi_agent = agents_dict["Kxi"]
liq_agent = agents_dict["Liq"]
kai_agent = agents_dict["Kai"]
vivi_agent = agents_dict["Vivi"]
nn_agent = agents_dict["Nn"]

# Define order for round-robin
DEBATER_AGENTS = [liq_agent, kai_agent, vivi_agent, nn_agent]

# --- UAGENTS Message Model ---
class Message(UagentsModel):
    text: str
    # Optional: Add speaker name if needed for context, but history has it
    # speaker: str

# Flag to control agent activity
conversation_active = False

# --- PanelManager Class Definition ---
class PanelManager:
    def __init__(self):
        logger.info("Initializing PanelManager...")
        self.status = "Idle"
        self.active = False
        self.history = []
        self.num_agents = len(DEBATER_AGENTS) # Number of debaters
        self.websockets = set()
        self.bureau_task = None # Task handle for Bureau
        self.bureau = None      # Initialize Bureau as None here

        # --- Initialize TTS ---
        self.tts = TTS()
        # --- End TTS Initialization ---

        # --- Agent Speaker and Prompt Maps ---
        self.agent_speaker_map = {agent["name"]: agent.get("speaker_id", 0) for agent in AGENTS_CONFIG}
        self.agent_prompt_map = {agent["name"]: agent["prompt"] for agent in AGENTS_CONFIG}
        # Agent addresses are globally defined, but maybe regenerate map in start? For now, keep.
        self.agent_address_map = {agent.name: agent.address for agent in agents_dict.values()}
        logger.info(f"Agent Speaker ID Map: {self.agent_speaker_map}")
        logger.info(f"Agent Addresses: {self.agent_address_map}")
        # --- End Maps ---

        # --- Conversation Control ---
        self.message_counter = 0
        self.max_messages = 12
        self.current_debater_index = 0

        # --- REMOVE Bureau Initialization from __init__ ---
        # bureau_port = 8005
        # bureau_endpoint = f"http://127.0.0.1:{bureau_port}/submit"
        # logger.info(f"Initializing Bureau with http server on port {bureau_port} and endpoint {bureau_endpoint}")
        # self.bureau = Bureau(port=bureau_port, endpoint=[bureau_endpoint])
        # for agent_name, agent_instance in agents_dict.items():
        #     logger.info(f"Adding agent {agent_name} (Address: {agent_instance.address}) to Bureau.")
        #     self.bureau.add(agent_instance)
        # --- End REMOVE ---

    # --- WebSocket Methods (unchanged) ---
    async def add_websocket(self, websocket, remote_addr: str | None):
        logger.info(f"Adding WebSocket connection from: {remote_addr or 'Unknown'}")
        self.websockets.add(websocket)
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})
        await self.broadcast_message({"type": "conversation_history", "payload": self.get_conversation_data()})

    def remove_websocket(self, websocket, remote_addr: str | None):
        logger.info(f"Removing WebSocket connection from: {remote_addr or 'Unknown'}")
        self.websockets.discard(websocket)

    async def broadcast_message(self, message_data: dict, exclude_sender=None):
        """Sends a JSON message to all connected WebSocket clients, optionally excluding one."""
        if not self.websockets:
            # logger.info("Broadcast: No active WebSocket clients.") # Can be noisy
            return

        logger.info(f"Broadcasting message to {len(self.websockets)} clients (excluding sender: {exclude_sender is not None}): {message_data}")
        message_json = json.dumps(message_data)
        tasks = []
        closed_sockets = []

        for ws in self.websockets:
            if ws == exclude_sender: continue # Skip the excluded sender
            if not ws.closed:
                tasks.append(ws.send_str(message_json))
            else:
                closed_sockets.append(ws) # Collect closed sockets for removal later

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Optional: Log errors from gather results if needed
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Find corresponding websocket for logging, requires careful tracking if needed
                    logger.error(f"Error broadcasting message: {result}")

        # Clean up closed sockets outside the loop
        if closed_sockets:
            logger.info(f"Removing {len(closed_sockets)} closed WebSocket connections detected during broadcast.")
            for ws in closed_sockets:
                self.remove_websocket(ws, None) # Use the remove method

    # --- Data Getters (unchanged) ---
    def get_status_data(self):
        return {
            "status": self.status,
            "active": self.active,
            "num_agents": self.num_agents, # Number of debaters
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def get_conversation_data(self):
        return { "history": self.history }

    # --- Ollama Interaction (unchanged) ---
    async def get_ollama_response(self, agent_name: str, history: list, history_turns: int = 10) -> str:
        logger.info(f"--- (1) Getting Ollama response for {agent_name} ---")
        url = "http://127.0.0.1:11434/api/chat"
        personality_prompt = self.agent_prompt_map.get(agent_name, "You are a helpful assistant.")

        # --- Modify the System Prompt Instruction ---
        system_content = f"{personality_prompt}\n\nRespond in MAXIMUM 2 SENTENCES. Be casual and conversational. Use the conversation history for context."
        # --- End Modification ---

        messages = [{"role": "system", "content": system_content}]
        recent_history = history[-(history_turns):]
        for msg in recent_history:
             role = "assistant" if msg['agent'] == agent_name else "user"
             if msg['agent'] != 'System':
                 messages.append({"role": role, "content": msg['text']})

        logger.debug(f"Messages being sent to Ollama /api/chat:\n{json.dumps(messages, indent=2)}")

        payload = {
            "model": "llama3",
            "messages": messages,
            "stream": False
        }
        try:
            logger.info(f"--- (2) Preparing to send POST to Ollama at {url} ---")
            async with aiohttp.ClientSession() as session:
                logger.info(f"--- (3) Sending POST to Ollama CHAT NOW ---")
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response: # Increased timeout slightly
                    logger.info(f"--- (4) Received Ollama status: {response.status} ---")
                    if response.status == 200:
                        data = await response.json()
                        # --- Extract response from /api/chat structure ---
                        if 'message' in data and 'content' in data['message']:
                            response_text = data['message']['content']
                            logger.info(f"Raw response content from Ollama /api/chat: {response_text[:200]}...")
                            # Cleaning might still be useful if model adds extraneous text
                            cleaned_response = extract_conversation(response_text) # Use existing cleaning function
                            logger.info(f"Cleaned response: {cleaned_response[:100]}...")
                            logger.info(f"--- (5) EXITING get_ollama_response (Success) ---")
                            return cleaned_response
                        else:
                             logger.error(f"Ollama /api/chat response missing 'message.content': {data}")
                             return "Error: Invalid response structure from Ollama chat."
                        # --- End extraction ---
                    else:
                        error_text = await response.text()
                        error_msg = f"Ollama returned status {response.status}. Response: {error_text[:200]}..."
                        logger.error(error_msg)
                        logger.info(f"--- (5) EXITING get_ollama_response (Ollama Error Status {response.status}) ---")
                        return f"Error: Unable to get response from Ollama (Status {response.status})"
        except asyncio.TimeoutError:
             logger.error("--- (E1) EXITING get_ollama_response (Timeout Error) ---")
             return "Error: Ollama request timed out."
        except aiohttp.ClientConnectorError as e:
             logger.error(f"Ollama connection error for {agent_name}: {e}", exc_info=True)
             return f"Error: Cannot connect to Ollama at {url}. Please ensure Ollama is running. Details: {e}"
        except Exception as e:
             logger.error(f"--- (E2) EXITING get_ollama_response (Exception: {e}) ---", exc_info=True)
             return f"Error: {str(e)}"

    # --- Modified: Handle Agent Response (Called by Agent Handlers) ---
    async def handle_agent_response(self, agent_name: str, agent_address: str, text: str):
        """Adds agent response to history, broadcasts text, triggers TTS, checks limit."""
        logger.info(f"PanelManager handling response from {agent_name}: {text[:60]}...")

        # --- FIX: Check if conversation is active ---
        if not self.active:
            logger.warning(f"PanelManager received response from {agent_name} but panel is inactive. Skipping.")
            return False # Indicate no further action needed

        # --- FIX: Increment counter ONLY for debaters, not moderator ---
        is_moderator = agent_name == kxi_agent.name
        if not is_moderator:
            self.message_counter += 1
            logger.info(f"Debater Message Count: {self.message_counter}/{self.max_messages}")


        timestamp = datetime.utcnow().isoformat() + "Z"
        message_payload = {
            "agent": agent_name,
            "address": agent_address,
            "text": text,
            "timestamp": timestamp,
            "audioStatus": "generating",
            "audioUrl": None,
        }
        self.history.append(message_payload)

        # Determine message type for frontend
        message_type = "moderator_message" if is_moderator else "agent_message"
        await self.broadcast_message({"type": message_type, "payload": message_payload})

        # Trigger background audio generation
        logger.info(f"Creating background task for TTS generation for {timestamp}")
        asyncio.create_task(self.generate_and_broadcast_audio(message_payload))

        # --- FIX: Check limit AFTER processing debater response ---
        if not is_moderator and self.message_counter >= self.max_messages:
            logger.info(f"Reached message limit ({self.max_messages}). Stopping panel automatically.")
            # Use create_task to avoid blocking the agent handler
            asyncio.create_task(self.stop_panel())
            return False # Signal to agent handler not to send next message

        return True # Signal to agent handler to proceed (send next message)


    # --- Generate and Broadcast Audio (unchanged) ---
    async def generate_and_broadcast_audio(self, message_payload: dict):
        agent_name = message_payload["agent"]
        text = message_payload["text"]
        timestamp = message_payload["timestamp"]

        speaker_id = self.agent_speaker_map.get(agent_name, 0) # Get speaker ID from map
        logger.info(f"Generating audio via SesameTTS wrapper for msg [{timestamp}], speaker {speaker_id}...")

        mp3_filepath_str = self.tts.generate_audio(text,speaker_id,self.broadcast_message)

        if mp3_filepath_str:
            mp3_filepath = Path(mp3_filepath_str)
            # Convert absolute filepath to relative URL path for frontend
            try:
                # --- Create URL relative to the static serving directory ---
                static_dir = Path(data_dir()) / "static" # Base static dir
                relative_path = mp3_filepath.relative_to(static_dir)
                # Ensure forward slashes for URL, add leading slash
                audio_url = "/" + relative_path.as_posix()
                # --- End URL Creation ---
                logger.info(f"Audio generated successfully for [{timestamp}]: {audio_url}")
                update_payload = {"timestamp": timestamp, "audioStatus": "ready", "audioUrl": audio_url}
            except ValueError as e:
                 logger.error(f"Failed to create relative path for {mp3_filepath} relative to {static_dir}: {e}. Sending failed status.")
                 update_payload = {"timestamp": timestamp, "audioStatus": "failed"}
        else:
            logger.error(f"Audio generation failed for msg [{timestamp}]")
            update_payload = {"timestamp": timestamp, "audioStatus": "failed"}

        logger.info(f"Broadcasting audio update for msg [{timestamp}]: {update_payload}")
        await self.broadcast_message({"type": "audio_update", "payload": update_payload})
    # --- End Generate Audio ---

    # --- Control Methods (Modified for Bureau) ---
    async def start_panel(self):
        """Starts the uAgents Bureau and conversation."""
        global conversation_active
        if self.active:
            logger.warning("Start panel called but already active.")
            return False
        # Ensure clean state before starting
        if self.bureau_task and not self.bureau_task.done():
             logger.warning("Start panel called but bureau_task seems active. Stopping first.")
             await self.stop_panel()
        if self.bureau is not None:
             logger.warning("Start panel called but bureau instance exists. Resetting.")
             self.bureau = None # Ensure bureau is None before recreating


        logger.info("Starting panel: Initializing NEW uAgents Bureau instance...")
        self.active = True
        conversation_active = True # Set global flag for agent handlers
        self.status = "Starting Bureau..."
        self.message_counter = 0
        self.current_debater_index = 0

        # --- Reset History and Broadcast ---
        debater_names_str = ", ".join([agent.name for agent in DEBATER_AGENTS])
        moderator_name_str = kxi_agent.name
        intro_participants = f"Featuring Moderator {moderator_name_str} and Debaters: {debater_names_str}"
        self.history = [{
            "agent": "System", "address": "system",
            "text": f"AI Panel: Crypto's Future. {intro_participants}. (Limit: {self.max_messages} debater responses)",
            "timestamp": datetime.utcnow().isoformat() + "Z", "audioStatus": "failed"
        }]
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})
        await self.broadcast_message({"type": "conversation_history", "payload": self.get_conversation_data()})
        # --- End Reset ---


        # --- MOVE Bureau Initialization HERE ---
        bureau_port = 8005 # Use a different port for the bureau itself
        bureau_endpoint = f"http://127.0.0.1:{bureau_port}/submit"
        logger.info(f"Initializing NEW Bureau with http server on port {bureau_port} and endpoint {bureau_endpoint}")
        self.bureau = Bureau(port=bureau_port, endpoint=[bureau_endpoint])
        for agent_name, agent_instance in agents_dict.items():
            # Make sure agent instances are the global ones defined outside the class
            logger.info(f"Adding agent {agent_name} (Address: {agent_instance.address}) to NEW Bureau.")
            self.bureau.add(agent_instance)
        # --- End MOVE ---


        logger.info("Creating background task for Bureau run_async.")
        if self.bureau is None: # Should not happen after above init
             logger.error("CRITICAL: Bureau instance is None just before creating task.")
             return False

        # Create the task to run the *new* bureau instance
        self.bureau_task = asyncio.create_task(self.bureau.run_async())

        # Kxi agent's startup event *within this new bureau* should now fire correctly

        self.status = f"Panel Active ({self.num_agents} debaters)"
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})
        logger.info("Panel start sequence complete. New Bureau running asynchronously.")
        return True

    async def stop_panel(self):
        """Stops the uAgents Bureau and conversation."""
        global conversation_active
        if not self.active and (not self.bureau_task or self.bureau_task.done()):
            logger.warning("Stop panel called but panel not active and no active task.")
            return False

        logger.info("Stopping panel (uAgents)...")
        original_state_active = self.active
        self.active = False
        conversation_active = False

        if original_state_active and self.status != "Stopping Bureau...":
            self.status = "Stopping Bureau..."
            await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})

        # --- Stop Bureau Task ---
        bureau_stopped = False
        if self.bureau_task and not self.bureau_task.done():
            logger.info("Cancelling Bureau task...")
            self.bureau_task.cancel()
            try:
                await asyncio.wait_for(self.bureau_task, timeout=3.0)
                logger.info("Bureau task awaited successfully after cancellation.")
                bureau_stopped = True
            except asyncio.CancelledError:
                logger.info("Bureau task cancelled successfully.")
                bureau_stopped = True
            except asyncio.TimeoutError:
                 logger.warning("Timeout waiting for Bureau task to cancel. Task might not have stopped cleanly.")
                 bureau_stopped = True
            except Exception as e:
                logger.error(f"Exception while awaiting cancelled Bureau task: {e}", exc_info=True)
                bureau_stopped = True
        else:
            logger.info("No active Bureau task found to cancel or task already done.")
            bureau_stopped = True

        self.bureau_task = None
        # --- ADD BACK: Set Bureau instance to None after stopping ---
        logger.info("Setting Bureau instance to None.")
        self.bureau = None
        # --- End ADD BACK ---

        # ... (rest of stop_panel logic: append history, broadcast status/history/message) ...
        logger.info("Panel stopped sequence complete (uAgents).")
        return True


# --- Create the single PanelManager instance ---
panel_manager = PanelManager()
logger.info("Global PanelManager instance created.")

# --- UAGENTS Agent Handlers ---

# Function to clean response (can be global or method if needed elsewhere)
def extract_conversation(text: str) -> str:
    # Remove the unwanted prefix if present
    prefix_pattern = r"^(Here's my response:)\s*"
    cleaned = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE).strip() # Use re.sub and ignore case

    # Existing basic cleaning
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip() # Also strip after removing quotes

    # Add more cleaning if Ollama adds unwanted prefixes/suffixes
    return cleaned

# --- Moderator Startup Handler ---
@kxi_agent.on_event("startup")
async def kxi_startup(ctx: Context):
    # This runs when the bureau starts the Kxi agent
    logger.info(f"MODERATOR_STARTUP [{ctx.agent.name}]: Startup event triggered.") # Log start
    # Check if the panel is intended to be active (via panel_manager)
    if panel_manager.active:
        logger.info(f"MODERATOR_STARTUP [{ctx.agent.name}]: Panel active, attempting to get opening statement from Ollama...")
        # Get opening statement/question from Ollama
        opening_text = await panel_manager.get_ollama_response(ctx.agent.name, panel_manager.history)
        logger.info(f"MODERATOR_STARTUP [{ctx.agent.name}]: Received response from Ollama: '{opening_text[:60]}...'") # Log response

        if opening_text.startswith("Error:"):
             logger.error(f"MODERATOR_STARTUP [{ctx.agent.name}]: Failed to get opening statement from Ollama: {opening_text}")
             # FIX: Replace undefined function call with log
             # await panel_manager.handle_single_message("System", f"Error starting conversation: {opening_text}")
             logger.error(f"MODERATOR_STARTUP [{ctx.agent.name}]: Bypassing undefined 'handle_single_message'. Stopping panel due to Ollama error.")
             asyncio.create_task(panel_manager.stop_panel()) # Stop if moderator fails
             return

        # Handle moderator's own message (adds to history, broadcasts, triggers TTS)
        logger.info(f"MODERATOR_STARTUP [{ctx.agent.name}]: Attempting to handle moderator's own opening message...")
        proceed = await panel_manager.handle_agent_response(ctx.agent.name, ctx.agent.address, opening_text)
        logger.info(f"MODERATOR_STARTUP [{ctx.agent.name}]: handle_agent_response returned: {proceed}")

        if proceed and conversation_active:
             # Send the *opening text* as the first message to the first debater
             first_debater_address = panel_manager.agent_address_map[DEBATER_AGENTS[0].name]
             logger.info(f"MODERATOR_STARTUP [{ctx.agent.name}]: Attempting to send opening message to {DEBATER_AGENTS[0].name} ({first_debater_address})...")
             try:
                 await ctx.send(first_debater_address, Message(text=opening_text))
                 logger.info(f"MODERATOR_STARTUP [{ctx.agent.name}]: Successfully sent opening message to {DEBATER_AGENTS[0].name}.") # Log success
             except Exception as e:
                 logger.error(f"MODERATOR_STARTUP [{ctx.agent.name}]: Failed to send opening message to {DEBATER_AGENTS[0].name}: {e}", exc_info=True)
                 # Optionally broadcast error or stop panel here if needed later
        elif not proceed:
             logger.warning(f"MODERATOR_STARTUP [{ctx.agent.name}]: 'proceed' is False after handle_agent_response. Not sending to first debater.")
        elif not conversation_active:
            logger.warning(f"MODERATOR_STARTUP [{ctx.agent.name}]: 'conversation_active' is False after handle_agent_response. Not sending to first debater.")
    else:
        logger.info(f"MODERATOR_STARTUP [{ctx.agent.name}]: Startup event fired, but panel_manager is not active. No action taken.")


# --- Generic Debater Message Handler ---
async def handle_debater_message(ctx: Context, sender: str, msg: Message, next_agent: Agent):
    agent_name = ctx.agent.name
    # Log entry into the handler
    logger.info(f"DEBATER_HANDLER [{agent_name}]: Entered handle_debater_message. Received message from {sender}: '{msg.text[:50]}...'")
    if not conversation_active:
        logger.info(f"DEBATER_HANDLER [{agent_name}]: Conversation inactive, skipping processing.")
        return

    logger.info(f"DEBATER_HANDLER [{agent_name}]: Calling PanelManager.get_ollama_response...")
    response_text = await panel_manager.get_ollama_response(agent_name, panel_manager.history)

    if response_text.startswith("Error:"):
        logger.error(f"{agent_name} failed to get response: {response_text}")
        # Decide how to handle LLM errors - skip turn? broadcast system message?
        # For now, let's just log and potentially skip sending
        proceed = await panel_manager.handle_agent_response(agent_name, ctx.agent.address, response_text) # Still log the error message
        # Maybe don't proceed to next agent on error?
        proceed = False # Don't send error messages as triggers
    else:
        cleaned_response = extract_conversation(response_text)
        logger.info(f"DEBATER_HANDLER [{agent_name}]: Calling PanelManager.handle_agent_response...")
        # Handle response first (adds to history, broadcasts, triggers TTS, checks limit)
        proceed = await panel_manager.handle_agent_response(agent_name, ctx.agent.address, cleaned_response)

    # Check if the panel is *still* active and if handler allows proceeding
    if proceed and conversation_active:
        next_agent_address = panel_manager.agent_address_map[next_agent.name]
        logger.info(f"DEBATER_HANDLER [{agent_name}]: Sending response to {next_agent.name} ({next_agent_address}).")
        # Send the *cleaned response* to the next agent
        await ctx.send(next_agent_address, Message(text=cleaned_response))
    else:
        logger.info(f"DEBATER_HANDLER [{agent_name}]: Panel stopped or handler indicated stop. Not sending reply to {next_agent.name}.")


# --- Assign Handlers Dynamically ---
for i, current_agent in enumerate(DEBATER_AGENTS):
    next_agent_index = (i + 1) % len(DEBATER_AGENTS)
    next_agent = DEBATER_AGENTS[next_agent_index]

    # Need to use a closure or default argument to capture current_agent and next_agent
    def create_handler(agent_to_reply_to):
        async def specific_handler(ctx: Context, sender: str, msg: Message):
            await handle_debater_message(ctx, sender, msg, agent_to_reply_to)
        return specific_handler

    logger.info(f"Assigning message handler to {current_agent.name}, will reply to {next_agent.name}")
    current_agent.on_message(model=Message)(create_handler(next_agent))


# --- WebSocket Handler (Add More Robust Error Handling) ---
async def websocket_handler(request):
    # global panel_manager # No longer needed
    from aiohttp import web, WSMsgType # Keep local import if preferred

    remote_addr = request.remote
    logger.info(f"WS_HANDLER [{remote_addr}]: --- ENTERING websocket_handler ---")

    ws = web.WebSocketResponse()
    try:
        await ws.prepare(request)
        logger.info(f"WS_HANDLER [{remote_addr}]: WebSocket connection prepared.")
    except Exception as e:
        logger.error(f"WS_HANDLER [{remote_addr}]: Failed to prepare WebSocket connection: {e}", exc_info=True)
        # Cannot return ws if prepare failed, maybe raise or return None/error response if framework allows
        return web.Response(status=500, text="WebSocket preparation failed") # Example error return


    # Pass the address string to add_websocket
    await panel_manager.add_websocket(ws, remote_addr)
    logger.info(f"WS_HANDLER [{remote_addr}]: Added WebSocket to PanelManager.")

    try:
        # Loop to receive messages FROM the client (currently none expected)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                 data = msg.data
                 logger.info(f"WS_HANDLER [{remote_addr}]: Received TEXT message: {data}")
                 # Optional: Handle commands sent FROM frontend via WS if needed later
            elif msg.type == WSMsgType.BINARY:
                 logger.info(f"WS_HANDLER [{remote_addr}]: Received BINARY message (length: {len(msg.data)}).")
                 # Handle binary data if needed later
            elif msg.type == WSMsgType.ERROR:
                # This specifically catches WebSocket protocol errors
                logger.error(f'WS_HANDLER [{remote_addr}]: WebSocket connection closed with exception {ws.exception()}', exc_info=ws.exception())
                break # Exit loop on WebSocket error
            elif msg.type == WSMsgType.CLOSED:
                 logger.info(f"WS_HANDLER [{remote_addr}]: WebSocket CLOSED message received.")
                 break # Exit loop gracefully if client sends close frame
            # Handle other types like PING/PONG if necessary
            # elif msg.type == WSMsgType.PING:
            #     await ws.pong()
            # elif msg.type == WSMsgType.PONG:
            #     logger.debug(f"WS_HANDLER [{remote_addr}]: Pong received.")

    except asyncio.CancelledError:
         # Catch task cancellation specifically
         logger.info(f"WS_HANDLER [{remote_addr}]: WebSocket handler task cancelled.")
         # Should lead to finally block
    except Exception as e:
         # Catch other unexpected errors during the receive loop
         logger.error(f"WS_HANDLER [{remote_addr}]: Error during WebSocket receive loop: {e}", exc_info=True)
         # Should lead to finally block
    finally:
        # This block executes when the loop exits for ANY reason
        # (normal close, error, cancellation, break)
        logger.info(f'WS_HANDLER [{remote_addr}]: Entering finally block, connection closing...')
        await panel_manager.remove_websocket(ws, remote_addr) # Ensure removal from manager
        # Check if WS is already closing/closed before trying to close again
        if not ws.closed:
            logger.info(f"WS_HANDLER [{remote_addr}]: WebSocket not closed yet, attempting graceful close.")
            await ws.close(code=WSMsgType.CLOSE, message=b'Server shutdown')
            logger.info(f"WS_HANDLER [{remote_addr}]: WebSocket closed from finally block.")
        else:
             logger.info(f"WS_HANDLER [{remote_addr}]: WebSocket already closed.")
        logger.info(f"WS_HANDLER [{remote_addr}]: --- EXITING websocket_handler ---")

    return ws # Return the response object

# --- Removed __main__ block (should be run via app.py) ---