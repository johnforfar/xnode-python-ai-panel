import os
app_dir = os.path.dirname(__file__)
os.chdir(app_dir)
print(f"INFO: Changed working directory to: {os.getcwd()}")

import logging
import logging.handlers
from pathlib import Path
import asyncio
import time
import re
from datetime import datetime
import json
import aiohttp # Keep aiohttp for get_ollama_response

# --- Logging Setup (Keep as is) ---
PROJECT_ROOT = Path(app_dir).parent.parent
LOG_FILE = PROJECT_ROOT / "logs.txt"
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
logger.info("Application logger initialized.")

# --- Third-Party Imports (Removed FastAPI/Starlette) ---
import torch
import torchaudio
# from uagents import Agent, Context, Model # Keep if needed by panel logic, remove otherwise

# --- Application Imports ---
from sesame_tts import SesameTTS

# --- Constants / Config ---
AGENTS = [
    {"name": "Michael Saylor", "seed": "michaelsaylor_seed",
     "prompt": "You are Michael Saylor..."}, # Truncated for brevity
    {"name": "Elon Musk", "seed": "elonmusk_seed",
     "prompt": "You are Elon Musk..."}, # Truncated for brevity
]

# --- PanelManager Class Definition ---
class PanelManager:
    def __init__(self):
        logger.info("Initializing PanelManager...")
        self.status = "Idle"
        self.active = False
        self.conversation_running = False
        self.history = []
        self.num_agents = 0
        self.active_agents_config = []
        self.current_agent_index = 0
        self.websocket = None
        self.conversation_task = None

        # Initialize TTS
        try:
            logger.info(f"PanelManager attempting to initialize TTS with model dir: {PROJECT_ROOT / 'models'}")
            self.tts = SesameTTS(device="cpu", model_dir=str(PROJECT_ROOT / "models"))
            self.tts_available = self.tts.tts_available
            logger.info(f"PanelManager TTS Available: {self.tts_available}")
        except Exception as e:
            logger.error(f"PanelManager failed during TTS init: {e}", exc_info=True)
            self.tts = None
            self.tts_available = False

        # --- Add WebSocket Management INSIDE the manager ---
        self.websockets = set() # Store connected sockets here

    # --- Add WebSocket Management INSIDE the manager ---
    async def add_websocket(self, websocket, remote_addr: str | None):
        logger.info(f"Adding WebSocket connection from: {remote_addr or 'Unknown'}")
        self.websockets.add(websocket)
        await self.broadcast_message({
            "type": "status_update",
            "payload": self.get_status_data()
        }, exclude_sender=None)

    def remove_websocket(self, websocket, remote_addr: str | None):
        logger.info(f"Removing WebSocket connection from: {remote_addr or 'Unknown'}")
        self.websockets.discard(websocket)
        if not self.websockets and self.active:
             logger.warning("Last WebSocket disconnected while panel active.")
             # Optionally stop panel

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

    async def start_panel(self, num_agents_req: int):
        """Starts the panel discussion."""
        if self.active:
            logger.warning("Start command received but panel is already active.")
            return False # Already running

        logger.info(f"Starting panel with {num_agents_req} agents.")
        self.num_agents = num_agents_req
        self.active = True
        self.status = f"Panel starting with {self.num_agents} agents..."
        self.history = [{
            "agent": "System", "address": "system",
            "text": f"AI Panel discussion started with {self.num_agents} agents.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }]
        self.active_agents_config = []
        self.current_agent_index = 0

        for i in range(min(self.num_agents, len(AGENTS))):
             agent_config = AGENTS[i]
             logger.info(f"Adding Agent to active list: {agent_config['name']}")
             self.active_agents_config.append({"config": agent_config, "index": i})

        await asyncio.sleep(0.1)
        self.status = f"Panel active ({self.num_agents} agents)"
        self.conversation_running = True

        # *** Use broadcast_message for status updates ***
        await self.broadcast_message({
            "type": "status_update",
            "payload": self.get_status_data() # Send updated status
        })
        await self.broadcast_message({
            "type": "system_message",
            "payload": {"text": f"Panel starting with {self.num_agents} agents."}
        })

        # --- Modify conversation loop start ---
        if self.websockets: # Check if ANY websocket is connected
             logger.info(f"Panel started. Found {len(self.websockets)} WebSocket connections. Starting conversation loop.")
             # Pass self (the manager instance) to the loop function
             # Ensure only one loop runs
             if self.conversation_task and not self.conversation_task.done():
                 logger.warning("Conversation task already running. Not starting another.")
             else:
                 self.conversation_task = asyncio.create_task(run_conversation_loop(self)) # Pass manager only
        else:
            logger.warning("Panel started but no active websockets. Waiting for connection to run conversation.")
            self.status = "Panel ready (Waiting for WS connection)"
            # Update status via broadcast if websockets connect later? Handled by add_websocket maybe.

        logger.info(f"Panel start sequence complete. Status: {self.status}")
        return True

    async def stop_panel(self):
        """Stops the panel discussion."""
        if not self.active:
            logger.warning("Stop command received but panel is not active.")
            return False

        logger.info("Stopping panel...")
        self.active = False
        self.conversation_running = False
        self.status = "Panel stopping..."

        if self.conversation_task and not self.conversation_task.done():
            logger.info("Cancelling conversation loop task...")
            self.conversation_task.cancel()
            try: await self.conversation_task
            except asyncio.CancelledError: logger.info("Conversation loop task cancelled successfully.")
            except Exception as e: logger.error(f"Exception while awaiting cancelled task: {e}")
        self.conversation_task = None

        self.history.append({
            "agent": "System", "address": "system",
            "text": "AI Panel discussion stopped.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        self.status = "Idle"
        self.num_agents = 0
        self.active_agents_config = []

        # *** Use broadcast_message for status updates ***
        await self.broadcast_message({
             "type": "status_update",
             "payload": self.get_status_data() # Send updated idle status
        })
        await self.broadcast_message({
            "type": "system_message",
            "payload": {"text": "Panel stopped."}
        })

        logger.info("Panel stopped.")
        return True

    async def get_ollama_response(self, personality_prompt: str, message: str) -> str:
        # *** Add more logging ***
        logger.info(f"--- ENTERING get_ollama_response for prompt: {personality_prompt[:30]}... ---")
        url = "http://127.0.0.1:11434/api/generate" # Use 127.0.0.1
        system_prompt = "Respond in MAXIMUM 3 SENTENCES. Be casual and conversational."
        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": f"{system_prompt}\n\n{personality_prompt}\n\nQuestion/Context: {message}\n\nYour brief response (MAXIMUM 3 SENTENCES):",
            "stream": False
        }
        try:
            logger.info(f"PanelManager sending POST to Ollama at {url}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get("response", "Error: No response field in Ollama data")
                        logger.info(f"Raw response from Ollama: {response_text[:200]}...")
                        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                        common_thinking_starts = ["Okay, so", "Alright, so", "Hmm, okay,", "Thinking:", "Let me break this down.", "Let me think.", "Here's my thought process:"]
                        if cleaned_response == response_text:
                            sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
                            start_index = 0
                            for i, sentence in enumerate(sentences):
                                is_thinking = any(sentence.strip().startswith(phrase) for phrase in common_thinking_starts)
                                if not is_thinking: start_index = i; break
                            actual_reply = " ".join(sentences[start_index:]).strip()
                            if actual_reply: cleaned_response = actual_reply
                        cleaned_response = cleaned_response.strip()
                        if not cleaned_response:
                            logger.warning("Response cleaning resulted in empty string, returning raw response.")
                            cleaned_response = response_text.strip()
                        logger.info(f"Cleaned response: {cleaned_response[:100]}...")
                        logger.info(f"--- EXITING get_ollama_response (Success) ---")
                        return cleaned_response
                    else:
                        error_msg = f"Ollama returned status {response.status}"
                        logger.error(error_msg)
                        logger.info(f"--- EXITING get_ollama_response (Ollama Error Status {response.status}) ---")
                        return f"Error: Unable to get response from Ollama (Status {response.status})"
        except asyncio.TimeoutError:
             logger.error("--- EXITING get_ollama_response (Timeout Error) ---")
             return "Error: Ollama request timed out."
        except Exception as e:
             logger.error(f"--- EXITING get_ollama_response (Exception: {e}) ---", exc_info=True)
             return f"Error: {str(e)}"


    async def send_agent_response(self, agent_data, prompt: str, context: str):
        # *** Add logging and use broadcast ***
        agent_name = agent_data['config']['name']
        logger.info(f"--- ENTERING send_agent_response for {agent_name} ---")
        # Inform frontend that agent is thinking via BROADCAST
        await self.broadcast_message({
             "type": "status_update",
             "payload": {"status": f"{agent_name} is thinking...", "active": True, "num_agents": self.num_agents}
        })
        try:
            response_text = await self.get_ollama_response(agent_data['config']['prompt'], prompt)
            if response_text and not response_text.startswith("Error:"):
                logger.info(f"Agent {agent_name} got response: {response_text[:60]}...")
                message_payload = {
                    "agent": agent_name, "address": f"agent_{agent_data['index']}",
                    "text": response_text, "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                self.history.append(message_payload) # Add to internal history
                # Broadcast the actual message
                await self.broadcast_message({"type": "agent_message", "payload": message_payload})
                # --- TTS Logic (kept as is, maybe add broadcast for audio path?) ---
                # ... (If TTS is successful, potentially broadcast an 'audio_ready' message?)
                # await self.broadcast_message({"type": "audio_ready", "payload": {"agent": agent_name, "path": web_audio_path}})
                await self.broadcast_message({ # Reset status after response/TTS
                    "type": "status_update",
                    "payload": {"status": f"Panel active ({self.num_agents} agents)", "active": True, "num_agents": self.num_agents}
                })
                logger.info(f"--- EXITING send_agent_response ({agent_name}, Success) ---")
                return response_text
            else:
                # ... (Handle error, use broadcast for system message) ...
                await self.broadcast_message({"type": "system_message", "payload": {"text": f"System: {agent_name} did not provide a valid response."}})
                await self.broadcast_message({ # Reset status after error
                    "type": "status_update",
                    "payload": {"status": f"Panel active ({self.num_agents} agents)", "active": True, "num_agents": self.num_agents}
                })
                logger.info(f"--- EXITING send_agent_response ({agent_name}, Failed Response) ---")
                return None
        except Exception as e:
            # ... (Handle exception, use broadcast for system message) ...
            await self.broadcast_message({"type": "system_message", "payload": {"text": f"System: Error getting response from {agent_name}."}})
            await self.broadcast_message({ # Reset status after error
                "type": "status_update",
                "payload": {"status": f"Panel active ({self.num_agents} agents)", "active": True, "num_agents": self.num_agents}
            })
            logger.error(f"--- EXITING send_agent_response ({agent_name}, Exception: {e}) ---", exc_info=True)
            return None

# --- Create the single PanelManager instance ---
panel_manager = PanelManager()
logger.info("Global PanelManager instance created.")

# --- Conversation Loop ---
# *** Modify to take only manager, use manager.broadcast ***
async def run_conversation_loop(manager: PanelManager, max_turns=10):
    logger.info(f"--- ENTERING conversation loop for {manager.num_agents} agents ---")
    manager.status = "Conversation Running"
    last_message_text = manager.history[-1]['text'] if manager.history else "Please introduce yourself briefly."
    if not manager.history or manager.history[-1]['agent'] != "System":
        manager.history.append({ "agent":"System", "address":"system", "text": last_message_text, "timestamp": datetime.utcnow().isoformat() + "Z" })
    turn = 0
    current_agent_turn_index = 0
    while manager.conversation_running and turn < max_turns and manager.active_agents_config:
        turn += 1
        if not manager.active_agents_config: break
        current_agent_turn_index = current_agent_turn_index % len(manager.active_agents_config)
        agent_data = manager.active_agents_config[current_agent_turn_index]
        agent_name = agent_data['config']['name']
        logger.info(f"Conversation Turn {turn}: {agent_name}'s turn.")
        # *** Use manager.broadcast for status update ***
        await manager.broadcast_message({
             "type": "status_update",
             "payload": {"status": f"Waiting for {agent_name}...", "active": True, "num_agents": manager.num_agents}
        })
        context_prompt = f"Current conversation context (last 3 messages):\n"
        history_context = "\n".join([f"{msg['agent']}: {msg['text']}" for msg in manager.history[-3:]])
        context_prompt += history_context + f"\n\nYour turn, {agent_name}. Respond to the last message: '{last_message_text}'"
        response_text = await manager.send_agent_response(agent_data, context_prompt, last_message_text)
        if response_text: last_message_text = response_text
        else: last_message_text = f"(Agent {agent_name} failed to respond). {agent_name}, please try again or comment on the previous message: '{last_message_text}'"
        current_agent_turn_index += 1
        await asyncio.sleep(2) # Keep delay between turns
    logger.info(f"--- EXITING conversation loop. Reason: {'Stopped by manager' if not manager.conversation_running else 'Max turns reached'} ---")
    if manager.active: manager.status = f"Panel active ({manager.num_agents} agents)"
    # *** Use manager.broadcast for final status ***
    await manager.broadcast_message({
         "type": "status_update",
         "payload": {"status": "Conversation Ended", "active": manager.active, "num_agents": manager.num_agents}
    })
    manager.conversation_running = False


# --- WebSocket Endpoint Handler ---
# *** Modify to use PanelManager for adding/removing sockets ***
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