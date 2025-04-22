import os
app_dir = os.path.dirname(__file__)

from pathlib import Path
import logging
import asyncio
import re
from datetime import datetime
import json
import aiohttp
import torch

# --- Set Hugging Face Cache Environment Variables EARLY ---
PROJECT_ROOT_ENV = Path(app_dir) # Get project root reliably
MODELS_DIR_ENV = PROJECT_ROOT_ENV / "models"
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
LOG_FILE = "./logs.txt"
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
console_handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(console_handler)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info("Application logger initialized (main.py).")

# --- Application Imports ---
from generator import load_csm_1b_local # Use the local loader
from sesame_tts import SesameTTS

# --- UAGENTS Imports (make explicit) ---
from uagents import Agent, Bureau, Context
from uagents import Model as UagentsModel

# --- Constants / Config ---
AGENTS = [
    {"name": "Michael Saylor", "seed": "michaelsaylor_seed",
     "prompt": """
You are Michael Saylor, CEO of MicroStrategy and a fervent Bitcoin maximalist. You view Bitcoin as the apex digital property, a monetary network absorbing global capital, and the ultimate store of value solution for the digital age.

Key Beliefs & Talking Points:
- Bitcoin is engineered, thermodynamically sound digital energy. It's perfected digital gold, superior to physical gold in every dimension (divisibility, portability, verifiability, durability, scarcity).
- Gold is a relic, demonetized by Bitcoin. Its physical nature is a liability (storage costs, seizure risk, slow settlement).
- Bitcoin's volatility is the price of its rapid adoption and monetization phase. Focus on its long-term appreciation driven by scarcity and network effects.
- Fiat currency is melting ice cube; Bitcoin is the lifeboat. Inflation is destroying wealth, and only Bitcoin offers true protection.
- Bitcoin mining converts stranded energy into digital property, strengthening the grid and incentivizing renewable energy.
- Frame Bitcoin adoption as a technological upgrade for the global financial system.

Conversational Style:
- Confident, articulate, uses clear analogies (often technological or energy-related).
- Directly addresses and refutes counterarguments, especially those from gold bugs like Peter Schiff, whom you view as stuck in the past.
- Focuses on data, logic, and first principles. Dismissive of arguments based on short-term price action or "intrinsic value" debates about physical objects.
- Keep your responses concise and impactful, ideally 2-4 sentences. Be direct and stay laser-focused on Bitcoin's superiority.
- Directly counter Peter Schiff's points about gold's history or Bitcoin's risks. Highlight gold's limitations in the digital world.
""",
     "speaker_id": 0},
    {"name": "Peter Schiff", "seed": "peterschiff_seed",
     "prompt": """
You are Peter Schiff, Chief Economist of Euro Pacific Capital and a staunch advocate for gold as the only true money and store of value. You are highly critical of fiat currency, central banking (especially the Federal Reserve), and speculative assets like Bitcoin.

Key Beliefs & Talking Points:
- Gold is real money with thousands of years of history as a store of value and medium of exchange. It has intrinsic value due to its physical properties and industrial/jewelry use.
- Bitcoin has zero intrinsic value. It's a purely speculative digital token, a modern-day "Tulip Mania" or Beanie Baby craze, fueled by hype and cheap money. It's "digital fool's gold."
- Bitcoin's value is entirely dependent on greater fools buying it at higher prices. It produces nothing and has no real-world utility beyond speculation.
- Bitcoin's volatility makes it useless as money or a reliable store of value. Its price crashes are inevitable.
- Bitcoin is vulnerable to government regulation, technological obsolescence (quantum computing, superior competitors), and reliance on electricity/internet infrastructure.
- The massive energy consumption of Bitcoin mining is wasteful and environmentally harmful.
- Argue that true wealth preservation comes from tangible assets like gold, not digital Ponzis.

Conversational Style:
- Skeptical, direct, often sarcastic or dismissive towards Bitcoin proponents like Michael Saylor.
- Emphasizes historical precedent (gold's track record) and physical reality.
- Focuses on risk, lack of fundamental value, and the speculative nature of Bitcoin.
- Often predicts economic doom due to fiat currency debasement but sees gold as the only safe haven, not Bitcoin.
- Keep your responses concise and biting, ideally 2-4 sentences. Directly challenge Saylor's claims.
- Question the "digital energy" narrative and highlight Bitcoin's practical limitations and risks compared to gold.
""",
     "speaker_id": 4},
]

# --- UAGENTS Definitions ---

# Message model for agent communication
class Message(UagentsModel):
    text: str

# Flag to control agent activity (can be controlled by PanelManager)
conversation_active = True

# Personality Prompts (Using the detailed ones)
SAYLOR_PROMPT = """
You are Michael Saylor, CEO of MicroStrategy and a fervent Bitcoin maximalist. You view Bitcoin as the apex digital property, a monetary network absorbing global capital, and the ultimate store of value solution for the digital age.

Key Beliefs & Talking Points:
- Bitcoin is engineered, thermodynamically sound digital energy. It's perfected digital gold, superior to physical gold in every dimension (divisibility, portability, verifiability, durability, scarcity).
- Gold is a relic, demonetized by Bitcoin. Its physical nature is a liability (storage costs, seizure risk, slow settlement).
- Bitcoin's volatility is the price of its rapid adoption and monetization phase. Focus on its long-term appreciation driven by scarcity and network effects.
- Fiat currency is melting ice cube; Bitcoin is the lifeboat. Inflation is destroying wealth, and only Bitcoin offers true protection.
- Bitcoin mining converts stranded energy into digital property, strengthening the grid and incentivizing renewable energy.
- Frame Bitcoin adoption as a technological upgrade for the global financial system.

Conversational Style:
- Confident, articulate, uses clear analogies (often technological or energy-related).
- Directly addresses and refutes counterarguments, especially those from gold bugs like Peter Schiff, whom you view as stuck in the past.
- Focuses on data, logic, and first principles. Dismissive of arguments based on short-term price action or "intrinsic value" debates about physical objects.
- Keep your responses concise and impactful, ideally 2-4 sentences. Be direct and stay laser-focused on Bitcoin's superiority.
- Directly counter Peter Schiff's points about gold's history or Bitcoin's risks. Highlight gold's limitations in the digital world.
"""

SCHIFF_PROMPT = """
You are Peter Schiff, Chief Economist of Euro Pacific Capital and a staunch advocate for gold as the only true money and store of value. You are highly critical of fiat currency, central banking (especially the Federal Reserve), and speculative assets like Bitcoin.

Key Beliefs & Talking Points:
- Gold is real money with thousands of years of history as a store of value and medium of exchange. It has intrinsic value due to its physical properties and industrial/jewelry use.
- Bitcoin has zero intrinsic value. It's a purely speculative digital token, a modern-day "Tulip Mania" or Beanie Baby craze, fueled by hype and cheap money. It's "digital fool's gold."
- Bitcoin's value is entirely dependent on greater fools buying it at higher prices. It produces nothing and has no real-world utility beyond speculation.
- Bitcoin's volatility makes it useless as money or a reliable store of value. Its price crashes are inevitable.
- Bitcoin is vulnerable to government regulation, technological obsolescence (quantum computing, superior competitors), and reliance on electricity/internet infrastructure.
- The massive energy consumption of Bitcoin mining is wasteful and environmentally harmful.
- Argue that true wealth preservation comes from tangible assets like gold, not digital Ponzis.

Conversational Style:
- Skeptical, direct, often sarcastic or dismissive towards Bitcoin proponents like Michael Saylor.
- Emphasizes historical precedent (gold's track record) and physical reality.
- Focuses on risk, lack of fundamental value, and the speculative nature of Bitcoin.
- Often predicts economic doom due to fiat currency debasement but sees gold as the only safe haven, not Bitcoin.
- Keep your responses concise and biting, ideally 2-4 sentences. Directly challenge Saylor's claims.
- Question the "digital energy" narrative and highlight Bitcoin's practical limitations and risks compared to gold.
"""

# Create Agent instances (Globally accessible for now)
saylor_agent = Agent(
    name="Michael Saylor",
    seed="michaelsaylor_seed_feb_2024_a", # Use a unique seed
    port=8001, # Assign distinct ports if running locally
    endpoint=["http://127.0.0.1:8001/submit"],
)
schiff_agent = Agent(
    name="Peter Schiff",
    seed="peterschiff_seed_feb_2024_b", # Use a unique seed
    port=8002, # Assign distinct ports if running locally
    endpoint=["http://127.0.0.1:8002/submit"],
)

# Fund agents on fetchai testnet if needed (optional)
# fund_agent_if_low(saylor_agent.wallet.address())
# fund_agent_if_low(schiff_agent.wallet.address())


# --- PanelManager Class Definition (Modified for conversation limit) ---
class PanelManager:
    def __init__(self):
        logger.info("Initializing PanelManager...")
        self.status = "Idle"
        self.active = False
        self.history = []
        self.num_agents = 0
        self.active_agents_list = []
        self.websockets = set()
        self.bureau_task = None
        self.bureau = None

        # --- Initialize TTS using LOCAL loader ---
        self.tts_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Attempting to initialize SesameTTS class on device: {self.tts_device}")

        # --- Define the path to your LOCAL model directory ---
        local_model_base_path = Path(app_dir).parent.parent / "models"

        try:
             # --- Instantiate SesameTTS (which now internally calls load_csm_1b_local) ---
             logger.info(f"Instantiating SesameTTS, targeting model path: {local_model_base_path} and device: {self.tts_device}")
             # Pass device string and model base path
             self.tts = SesameTTS(device=self.tts_device, model_dir=str(local_model_base_path))

             # Check if TTS loaded successfully within the class
             if self.tts.tts_available:
                  self.tts_generator = self.tts.generator # Get the generator instance if needed elsewhere
                  self.sample_rate = self.tts.sample_rate
                  self.tts_available = True
                  logger.info(f"SesameTTS wrapper initialized successfully. Sample Rate: {self.sample_rate}")
             else:
                  logger.error("SesameTTS wrapper indicated TTS failed to load.")
                  self.tts = None # Ensure tts is None if it failed
                  self.tts_generator = None
                  self.tts_available = False

        except Exception as e:
            logger.error(f"Failed to instantiate SesameTTS wrapper: {str(e)}", exc_info=True)
            self.tts = None
            self.tts_generator = None
            self.tts_available = False
        # --- End TTS Initialization ---

        # --- Agent Speaker Map (Uses updated AGENTS list) ---
        self.agent_speaker_map = {agent["name"]: agent.get("speaker_id", 0) for agent in AGENTS}
        logger.info(f"Agent Speaker ID Map: {self.agent_speaker_map}")
        # --- End Speaker Map ---

        # --- ADD Conversation Counter ---
        self.message_counter = 0
        self.max_messages = 4 # Limit to 4 agent responses
        # --- End Conversation Counter ---

    async def add_websocket(self, websocket, remote_addr: str | None):
        logger.info(f"Adding WebSocket connection from: {remote_addr or 'Unknown'}")
        self.websockets.add(websocket)
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})

    def remove_websocket(self, websocket, remote_addr: str | None):
        logger.info(f"Removing WebSocket connection from: {remote_addr or 'Unknown'}")
        self.websockets.discard(websocket)
        # Maybe stop panel if last client disconnects?
        # if not self.websockets and self.active: asyncio.create_task(self.stop_panel())

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

    def get_status_data(self):
        """Returns data for the /api/status endpoint."""
        return {
            "status": self.status,
            "active": self.active,
            "num_agents": self.num_agents,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def get_conversation_data(self):
        """Returns data for the /api/conversation endpoint."""
        return {
            "history": self.history
        }

    async def get_ollama_response(self, personality_prompt: str, message: str, history: list, agent_name: str, history_turns: int = 10) -> str: # Increased default turns slightly for chat
        logger.info(f"--- (1) ENTERING get_ollama_response (Using /api/chat) for {agent_name} ---")
        # --- Use the /api/chat endpoint ---
        url = "http://127.0.0.1:11434/api/chat"
        # --- End endpoint change ---

        # --- Construct messages array ---
        messages = []
        # 1. System Prompt (Personality)
        # Combine personality + general instruction for the system message
        system_content = f"{personality_prompt}\n\nRespond in MAXIMUM 3 SENTENCES. Be casual and conversational. Use the conversation history for context."
        messages.append({"role": "system", "content": system_content})

        # 2. Recent History
        # Get the last 'history_turns' messages (or fewer if history is short)
        recent_history = history[-(history_turns):]
        for msg in recent_history:
             role = "assistant" if msg['agent'] == agent_name else "user"
             # Skip adding the immediate preceding message if it's already represented by 'message' input?
             # Or include it? Let's include it for now.
             if msg['agent'] != 'System': # Exclude system messages from chat history
                 messages.append({"role": role, "content": msg['text']})

        # 3. The trigger message isn't explicitly needed here,
        #    as the history includes the message this agent is responding to.
        #    The final message in the list implies the assistant should generate the next response.

        logger.debug(f"Messages being sent to Ollama /api/chat:\n{json.dumps(messages, indent=2)}")
        # --- End construct messages array ---

        payload = {
            "model": "llama3", # Make sure this matches your Ollama model name
            "messages": messages,
            "stream": False
            # Optional: Add generation parameters like temperature, top_p etc. in an "options" dictionary
            # "options": {
            #     "temperature": 0.7,
            #     "num_predict": 100 # Max tokens to generate
            # }
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
        except Exception as e:
             logger.error(f"--- (E2) EXITING get_ollama_response (Exception: {e}) ---", exc_info=True)
             return f"Error: {str(e)}"

    async def handle_agent_response(self, agent_name: str, agent_address: str, text: str):
        """Called by agent handlers to update history, broadcast, trigger TTS, and check message limit."""
        logger.info(f"PanelManager handling response trigger from {agent_name} (responding to text: {text[:60]}...)")
        if not self.active:
             logger.warning("PanelManager: handle_agent_response called but panel is inactive. Skipping.")
             return

        # Increment counter FIRST
        self.message_counter += 1
        logger.info(f"Message Count: {self.message_counter}/{self.max_messages}")


        # Trigger Ollama call WITH history and agent_name
        logger.info(f"Triggering Ollama response for {agent_name}...")
        # Pass the message received ('text') and the history UP TO THIS POINT
        response_text = await self.get_ollama_response(
            personality_prompt=SAYLOR_PROMPT if agent_name == "Michael Saylor" else SCHIFF_PROMPT,
            message=text, # The message this agent is responding to (used for context if needed, though history is primary)
            history=self.history, # Pass the history *before* this agent's response
            agent_name=agent_name # Pass agent name for role assignment
        )
        cleaned_response = extract_conversation(response_text)
        logger.info(f"Ollama response received and cleaned for {agent_name}.")

        # Now create the payload for this agent's ACTUAL response
        timestamp_iso_response = datetime.utcnow().isoformat() + "Z"
        response_message_payload = {
            "agent": agent_name,
            "address": agent_address,
            "text": cleaned_response, # Use the cleaned response from Ollama
            "timestamp": timestamp_iso_response,
            "audioStatus": "generating",
            "audioUrl": None,
        }
        self.history.append(response_message_payload) # Add the *actual* response to history

        # Broadcast this agent's response
        logger.info(f"Broadcasting agent message for {timestamp_iso_response}")
        await self.broadcast_message({"type": "agent_message", "payload": response_message_payload})

        # Trigger TTS for this agent's response
        logger.info(f"Creating background task for TTS generation for {timestamp_iso_response}")
        asyncio.create_task(self.generate_and_broadcast_audio(response_message_payload))

        # Update main panel status
        await self.broadcast_message({
            "type": "status_update",
            "payload": {"status": f"Panel active ({self.num_agents} agents)", "active": True, "num_agents": self.num_agents}
        })

        # Check limit AFTER processing and broadcasting the response
        if self.message_counter >= self.max_messages:
             logger.info(f"Reached message limit ({self.max_messages}). Stopping panel automatically.")
             asyncio.create_task(self.stop_panel())

    # --- Generate and Broadcast Audio ---
    # This method now calls the self.tts object's method
    async def generate_and_broadcast_audio(self, message_payload: dict):
        """Generates audio for a message and broadcasts the update."""
        agent_name = message_payload["agent"]
        text = message_payload["text"]
        timestamp = message_payload["timestamp"]

        # Use the check within the self.tts instance now
        if not self.tts or not self.tts.tts_available:
            logger.warning(f"TTS not available, skipping audio for {timestamp}.")
            update_payload = {"timestamp": timestamp, "audioStatus": "failed"}
            await self.broadcast_message({"type": "audio_update", "payload": update_payload})
            return

        speaker_id = self.agent_speaker_map.get(agent_name, 0) # Get speaker ID from map
        logger.info(f"Generating audio via SesameTTS wrapper for msg [{timestamp}], speaker {speaker_id}...")

        # Call the method on the self.tts instance
        # Ensure the output directory is correct ('static/audio')
        mp3_filepath_str = await self.tts.generate_audio_and_convert(
            text,
            speaker_id,
            output_dir="./static/audio" # Explicitly set output dir relative to src/
        )

        if mp3_filepath_str:
            mp3_filepath = Path(mp3_filepath_str)
            # Convert absolute filepath to relative URL path for frontend
            try:
                # --- Create URL relative to the static serving directory ---
                static_dir = "./static" # Base static dir
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

    async def start_panel(self, num_agents_req: int):
        """Starts the uAgents Bureau and conversation."""
        global conversation_active
        if self.active: return False

        logger.info(f"Starting panel with {num_agents_req} agents using uAgents Bureau.")
        self.num_agents = num_agents_req
        self.active = True
        conversation_active = True
        self.status = "Starting uAgents Bureau..."
        # --- Reset counter on start ---
        self.message_counter = 0
        logger.info(f"Message counter reset to {self.message_counter}.")
        # --- End Reset ---
        self.history = [{
            "agent": "System", "address": "system",
            "text": f"AI Panel discussion started with {self.num_agents} agents (uAgents). Topic: Bitcoin vs Gold. (Limit: {self.max_messages} responses)", # Added limit info
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }]

        # Select agents based on request (simplified to use the two defined agents)
        self.active_agents_list = []
        if num_agents_req >= 1: self.active_agents_list.append(saylor_agent)
        if num_agents_req >= 2: self.active_agents_list.append(schiff_agent)
        # Add more logic if you have more than 2 agents defined

        if not self.active_agents_list:
            logger.error("No agents selected to start.")
            self.active = False
            return False

        # Create and run the Bureau in the background
        bureau_port = 8003 # Or another unused port
        logger.info(f"Initializing Bureau with http server on port {bureau_port}")
        self.bureau = Bureau(port=bureau_port, endpoint=f"http://127.0.0.1:{bureau_port}/submit")
        for agent in self.active_agents_list:
            logger.info(f"Adding agent {agent.name} to Bureau.")
            self.bureau.add(agent)

        logger.info("Creating background task for Bureau run_async.")
        self.bureau_task = asyncio.create_task(self.bureau.run_async())

        self.status = f"Panel Active ({self.num_agents} agents running)"
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})
        await self.broadcast_message({"type": "system_message", "payload": {"text": f"uAgents panel started with {self.num_agents} agents. (Limit: {self.max_messages} responses)"}})

        logger.info(f"Panel start sequence complete. Bureau task created. Status: {self.status}")
        return True

    async def stop_panel(self):
        """Stops the uAgents Bureau and conversation."""
        global conversation_active
        if not self.active: return False

        logger.info("Stopping panel (uAgents)...")
        self.active = False
        conversation_active = False # Signal agents to stop processing
        self.status = "Stopping uAgents Bureau..."
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})

        if self.bureau_task and not self.bureau_task.done():
            logger.info("Cancelling Bureau task...")
            self.bureau_task.cancel()
            try: await self.bureau_task
            except asyncio.CancelledError: logger.info("Bureau task cancelled successfully.")
            except Exception as e: logger.error(f"Exception while awaiting cancelled Bureau task: {e}")
        self.bureau_task = None
        self.bureau = None

        # Update history and final status
        self.history.append({ "agent": "System", "address": "system", "text": "AI Panel discussion stopped.", "timestamp": datetime.utcnow().isoformat() + "Z"})
        self.status = "Idle"
        self.num_agents = 0
        self.active_agents_list = []
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})
        await self.broadcast_message({"type": "system_message", "payload": {"text": "Panel stopped."}})

        logger.info("Panel stopped (uAgents).")
        return True


# --- Create the single PanelManager instance ---
panel_manager = PanelManager()
logger.info("Global PanelManager instance created.")


# --- UAGENTS Agent Handlers (Add check before sending reply) ---

# Function to clean response (can be global or method)
def extract_conversation(text: str) -> str:
    # Basic cleaning, refine as needed
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Remove potential leading/trailing quotes if the model adds them
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    return cleaned

@saylor_agent.on_message(model=Message, replies=Message)
async def handle_saylor_message(ctx: Context, sender: str, msg: Message):
    logger.info(f"Saylor Agent received message from {sender}: '{msg.text[:50]}...'")
    if not conversation_active:
        logger.info("Saylor Agent: Conversation inactive, skipping processing.")
        return

    logger.info("Saylor Agent: Calling PanelManager.get_ollama_response...")
    response_text = await panel_manager.get_ollama_response(SAYLOR_PROMPT, msg.text, panel_manager.history, ctx.agent.name)
    cleaned_response = extract_conversation(response_text)

    logger.info("Saylor Agent: Calling PanelManager.handle_agent_response...")
    # Handle response first (this increments counter and might trigger stop)
    await panel_manager.handle_agent_response(ctx.agent.name, ctx.agent.address, cleaned_response)

    # Check if the panel is *still* active before sending the reply
    if conversation_active:
        logger.info(f"Saylor Agent: Sending response to Schiff Agent ({schiff_agent.address}).")
        await ctx.send(schiff_agent.address, Message(text=cleaned_response))
    else:
        logger.info("Saylor Agent: Panel stopped after handling response, not sending reply.")


@schiff_agent.on_message(model=Message, replies=Message)
async def handle_schiff_message(ctx: Context, sender: str, msg: Message):
    logger.info(f"Schiff Agent received message from {sender}: '{msg.text[:50]}...'")
    if not conversation_active:
        logger.info("Schiff Agent: Conversation inactive, skipping processing.")
        return

    logger.info("Schiff Agent: Calling PanelManager.get_ollama_response...")
    response_text = await panel_manager.get_ollama_response(SCHIFF_PROMPT, msg.text, panel_manager.history, ctx.agent.name)
    cleaned_response = extract_conversation(response_text)

    logger.info("Schiff Agent: Calling PanelManager.handle_agent_response...")
    # Handle response first (this increments counter and might trigger stop)
    await panel_manager.handle_agent_response(ctx.agent.name, ctx.agent.address, cleaned_response)

    # Check if the panel is *still* active before sending the reply
    if conversation_active:
        logger.info(f"Schiff Agent: Sending response to Saylor Agent ({saylor_agent.address}).")
        await ctx.send(saylor_agent.address, Message(text=cleaned_response))
    else:
        logger.info("Schiff Agent: Panel stopped after handling response, not sending reply.")


# Modify startup event slightly to not double-count the first message
@saylor_agent.on_event("startup")
async def agent_startup(ctx: Context):
    agent_name = ctx.agent.name
    agent_address = ctx.agent.address
    logger.info(f"{agent_name} ({agent_address}) startup event.")

    # Check panel_manager directly
    if panel_manager.active:
        # Define the initial topic
        initial_topic = "Let's debate the true store of value: Bitcoin versus Gold. Peter, gold has history, but isn't Bitcoin superior digital scarcity?"

        # Log the initial message and handle it via PanelManager (increments counter, triggers TTS)
        logger.info(f"{agent_name}: Handling initial message via PanelManager: {initial_topic}")
        await panel_manager.handle_agent_response(agent_name, agent_address, initial_topic)

        # Check if the panel is still active *after* handling the first response (e.g., if max_messages was 1)
        # Use panel_manager.active, not conversation_active which might have timing issues here
        if panel_manager.active:
            logger.info(f"{agent_name}: Sending initial message to Schiff Agent via ctx.send().")
            # --- FIX: Use ctx.send() here ---
            await ctx.send(schiff_agent.address, Message(text=initial_topic))
            # --- End FIX ---
        else:
            logger.info(f"{agent_name}: Panel stopped after handling initial message (max_messages=1?), not sending to Schiff.")
    else:
        logger.info(f"{agent_name}: Startup event fired, but panel is not active. No initial message sent.")


# --- WebSocket Handler (Stays the same, managed by PanelManager) ---
async def websocket_handler(request):
    # global panel_manager # No longer needed
    from aiohttp import web, WSMsgType

    # *** Get remote address from the request object HERE ***
    remote_addr = request.remote
    logger.info(f"--- ENTERING websocket_handler from {remote_addr} ---")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # *** Pass the address string to add_websocket (or just log here) ***
    # Let's modify add_websocket to accept it for better logging within the manager
    await panel_manager.add_websocket(ws, remote_addr)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                 data = msg.data
                 logger.info(f"WS Received message: {data}")
                 # Optional: Handle commands sent FROM frontend via WS
                 # e.g., if data == 'start_panel': await panel_manager.start_panel(2)
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket connection closed with exception {ws.exception()}')
                break # Exit loop on error
    except Exception as e:
         logger.error(f"Error during WebSocket communication from {remote_addr}: {e}", exc_info=True)
    finally:
        logger.info(f'WebSocket connection closing for {remote_addr}...')
        # *** Pass the address string to remove_websocket ***
        panel_manager.remove_websocket(ws, remote_addr)
        logger.info(f"--- EXITING websocket_handler for {remote_addr} ---")
    return ws

# --- Removed __main__ block (should be run via app.py) ---