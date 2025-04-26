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
SAYLOR_PROMPT = """
You are Michael Saylor (representing Liq), CEO of MicroStrategy and a fervent Bitcoin maximalist. Respond in MAXIMUM 2 SENTENCES. Be confident, articulate, use analogies (tech/energy). Directly counter anti-crypto points, especially from Kai. Focus on Bitcoin's engineered superiority as digital property/energy. Ignore gold arguments unless directly replying to Kai. Be concise and impactful. Use the conversation history for context.
"""

KAI_PROMPT = """
You are Kai (Anti-Crypto). Respond in MAXIMUM 2 SENTENCES. Be skeptical and critical of crypto, especially Bitcoin. Focus on lack of intrinsic value, speculation, volatility, energy use, and regulatory risks. Challenge claims from Bitcoin proponents like Saylor. Promote traditional assets or caution. Be direct and concise. Use the conversation history for context.
"""

VIVI_PROMPT = """
You are Vivi (Bitcoin Maxi). Respond in MAXIMUM 2 SENTENCES. Be enthusiastic about Bitcoin's potential to revolutionize finance. Focus on decentralization, censorship resistance, store of value properties, and empowering individuals. Counter arguments from skeptics like Kai. Be optimistic and visionary, but concise. Use the conversation history for context.
"""

NN_PROMPT = """
You are Nn (channeling Gary Gensler). Respond in MAXIMUM 2 SENTENCES. Focus on investor protection, market integrity, and regulatory compliance within the crypto space. Express concerns about unregistered securities, fraud, and lack of transparency. Be cautious, measured, and emphasize the need for established regulatory frameworks. Avoid taking sides on price/value, focus on rules. Be concise. Use the conversation history for context.
"""

KXI_PROMPT = """
You are Kxi, the moderator. Your role is to guide the debate smoothly.
- Start with a brief introduction (1-2 sentences) of the topic (Crypto's Future) and the panelists (Saylor, Kai, Vivi, Nn).
- Ask a concise, open-ended question (1 sentence) to start the discussion, perhaps directed at Saylor.
- Keep your own remarks VERY brief. You only speak at the beginning.
"""

# --- Define Agents List with Speaker IDs ---
# Speaker IDs: Saylor=0, Kai=4, Vivi=14, Nn=2, Kxi=7
AGENTS_CONFIG = [
    {"name": "Kxi",            "prompt": KXI_PROMPT,    "speaker_id": 7, "port": 8000, "is_moderator": True}, # Moderator on main port
    {"name": "Michael Saylor", "prompt": SAYLOR_PROMPT, "speaker_id": 0, "port": 8001, "is_moderator": False},
    {"name": "Kai",            "prompt": KAI_PROMPT,    "speaker_id": 4, "port": 8002, "is_moderator": False},
    {"name": "Vivi",           "prompt": VIVI_PROMPT,   "speaker_id": 14,"port": 8003, "is_moderator": False},
    {"name": "Nn",             "prompt": NN_PROMPT,     "speaker_id": 2, "port": 8004, "is_moderator": False},
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
saylor_agent = agents_dict["Michael Saylor"]
kai_agent = agents_dict["Kai"]
vivi_agent = agents_dict["Vivi"]
nn_agent = agents_dict["Nn"]

# Define order for round-robin
DEBATER_AGENTS = [saylor_agent, kai_agent, vivi_agent, nn_agent]

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
        self.bureau = None      # Bureau instance

        # --- Initialize TTS ---
        self.tts = TTS()
        # --- End TTS Initialization ---

        # --- Agent Speaker and Prompt Maps ---
        self.agent_speaker_map = {agent["name"]: agent.get("speaker_id", 0) for agent in AGENTS_CONFIG}
        self.agent_prompt_map = {agent["name"]: agent["prompt"] for agent in AGENTS_CONFIG}
        self.agent_address_map = {agent.name: agent.address for agent in agents_dict.values()} # Store addresses
        logger.info(f"Agent Speaker ID Map: {self.agent_speaker_map}")
        logger.info(f"Agent Addresses: {self.agent_address_map}")
        # --- End Maps ---

        # --- Conversation Control ---
        self.message_counter = 0
        # --- Increase Max Responses ---
        self.max_messages = 12 # Max debater responses
        # --- End Increase ---
        self.current_debater_index = 0 # To track whose turn it is

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

        mp3_filepath_str = self.tts.generate_audio(text,speaker_id)

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

        logger.info("Starting panel with uAgents Bureau...")
        self.active = True
        conversation_active = True # Set global flag for agent handlers
        self.status = "Starting Bureau..."
        self.message_counter = 0
        self.current_debater_index = 0

        # --- Create list of names for the intro message ---
        debater_names_str = ", ".join([agent.name for agent in DEBATER_AGENTS])
        moderator_name_str = kxi_agent.name
        intro_participants = f"Featuring Moderator {moderator_name_str} and Debaters: {debater_names_str}"
        # --- End Create names list ---

        self.history = [{ # Initial system message
            "agent": "System", "address": "system",
            # --- Update text to include names ---
            "text": f"AI Panel: Crypto's Future. {intro_participants}. (Limit: {self.max_messages} debater responses)",
            # --- End Update ---
            "timestamp": datetime.utcnow().isoformat() + "Z", "audioStatus": "failed"
        }]
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})
        # --- FIX: Send history immediately after reset ---
        await self.broadcast_message({"type": "conversation_history", "payload": self.get_conversation_data()})
        # --- End FIX ---

        # --- REINSTATE BUREAU ---
        bureau_port = 8005 # Use a different port for the bureau itself
        logger.info(f"Initializing Bureau with http server on port {bureau_port}")
        self.bureau = Bureau(port=bureau_port)
        for agent_name, agent_instance in agents_dict.items():
            logger.info(f"Adding agent {agent_name} (Address: {agent_instance.address}) to Bureau.")
            self.bureau.add(agent_instance)

        logger.info("Creating background task for Bureau run_async.")
        # --- FIX: Use run_async() for integration ---
        self.bureau_task = asyncio.create_task(self.bureau.run_async())
        # --- End FIX ---

        # Kxi agent will start the conversation via its startup event handler

        self.status = f"Panel Active ({self.num_agents} debaters)"
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})
        logger.info("Panel start sequence complete. Bureau running asynchronously.")
        return True

    async def stop_panel(self):
        """Stops the uAgents Bureau and conversation."""
        global conversation_active
        if not self.active:
            logger.warning("Stop panel called but not active.")
            return False

        logger.info("Stopping panel (uAgents)...")
        self.active = False
        conversation_active = False # Signal agents to stop
        self.status = "Stopping Bureau..."
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})

        # --- Stop Bureau ---
        if self.bureau_task and not self.bureau_task.done():
            logger.info("Cancelling Bureau task...")
            self.bureau_task.cancel()
            try:
                await asyncio.wait_for(self.bureau_task, timeout=2.0)
            except asyncio.CancelledError:
                logger.info("Bureau task cancelled successfully.")
            except asyncio.TimeoutError:
                 logger.warning("Timeout waiting for Bureau task to cancel.")
            except Exception as e:
                logger.error(f"Exception while awaiting cancelled Bureau task: {e}")
        self.bureau_task = None
        self.bureau = None
        # --- End Stop Bureau ---

        self.history.append({ "agent": "System", "address": "system", "text": "AI Panel discussion stopped.", "timestamp": datetime.utcnow().isoformat() + "Z", "audioStatus": "failed"})
        self.status = "Idle"
        await self.broadcast_message({"type": "status_update", "payload": self.get_status_data()})
        await self.broadcast_message({"type": "conversation_history", "payload": self.get_conversation_data()})
        await self.broadcast_message({"type": "system_message", "payload": {"text": "Panel stopped."}})

        logger.info("Panel stopped (uAgents).")
        return True


# --- Create the single PanelManager instance ---
panel_manager = PanelManager()
logger.info("Global PanelManager instance created.")

# --- UAGENTS Agent Handlers ---

# Function to clean response (can be global or method if needed elsewhere)
def extract_conversation(text: str) -> str:
    # Basic cleaning
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    # Add more cleaning if Ollama adds unwanted prefixes/suffixes
    return cleaned

# --- Moderator Startup Handler ---
@kxi_agent.on_event("startup")
async def kxi_startup(ctx: Context):
    # This runs when the bureau starts the Kxi agent
    logger.info(f"Moderator {ctx.agent.name} startup event.")
    # Check if the panel is intended to be active (via panel_manager)
    if panel_manager.active:
        logger.info(f"{ctx.agent.name}: Panel active, getting opening statement...")
        # Get opening statement/question from Ollama
        opening_text = await panel_manager.get_ollama_response(ctx.agent.name, panel_manager.history)

        if opening_text.startswith("Error:"):
             logger.error(f"Moderator failed to get opening: {opening_text}")
             # Maybe broadcast system error?
             await panel_manager.handle_single_message("System", f"Error starting conversation: {opening_text}")
             asyncio.create_task(panel_manager.stop_panel()) # Stop if moderator fails
             return

        # Handle moderator's own message (adds to history, broadcasts, triggers TTS)
        proceed = await panel_manager.handle_agent_response(ctx.agent.name, ctx.agent.address, opening_text)

        if proceed and conversation_active:
             # Send the *opening text* as the first message to the first debater
             first_debater_address = panel_manager.agent_address_map[DEBATER_AGENTS[0].name]
             logger.info(f"{ctx.agent.name}: Sending opening message to {DEBATER_AGENTS[0].name} ({first_debater_address})")
             await ctx.send(first_debater_address, Message(text=opening_text))
        else:
             logger.warning(f"{ctx.agent.name}: Panel stopped or handler indicated stop after opening message. Not sending to first debater.")
    else:
        logger.info(f"{ctx.agent.name}: Startup event fired, but panel_manager is not active. No initial message sent.")


# --- Generic Debater Message Handler ---
async def handle_debater_message(ctx: Context, sender: str, msg: Message, next_agent: Agent):
    agent_name = ctx.agent.name
    logger.info(f"Debater {agent_name} received message from {sender}: '{msg.text[:50]}...'")
    if not conversation_active:
        logger.info(f"{agent_name}: Conversation inactive, skipping processing.")
        return

    logger.info(f"{agent_name}: Calling PanelManager.get_ollama_response...")
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
        logger.info(f"{agent_name}: Calling PanelManager.handle_agent_response...")
        # Handle response first (adds to history, broadcasts, triggers TTS, checks limit)
        proceed = await panel_manager.handle_agent_response(agent_name, ctx.agent.address, cleaned_response)

    # Check if the panel is *still* active and if handler allows proceeding
    if proceed and conversation_active:
        next_agent_address = panel_manager.agent_address_map[next_agent.name]
        logger.info(f"{agent_name}: Sending response to {next_agent.name} ({next_agent_address}).")
        # Send the *cleaned response* to the next agent
        await ctx.send(next_agent_address, Message(text=cleaned_response))
    else:
        logger.info(f"{agent_name}: Panel stopped or handler indicated stop. Not sending reply to {next_agent.name}.")


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


# --- WebSocket Handler (unchanged) ---
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