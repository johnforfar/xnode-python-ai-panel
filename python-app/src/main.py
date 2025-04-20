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

# --- PanelManager Class Definition (Keep as defined before) ---
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

        if self.websocket:
             logger.info("Creating background task for conversation loop.")
             # Pass self (the manager instance) to the loop function
             self.conversation_task = asyncio.create_task(run_conversation_loop(self.websocket, self))
        else:
            logger.warning("Panel started but no active websocket to run conversation.")
            self.status = "Panel ready (Waiting for WS connection)"

        logger.info(f"Panel started. Status: {self.status}")
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
        logger.info("Panel stopped.")
        return True

    async def get_ollama_response(self, personality_prompt: str, message: str) -> str:
        # --- (Ollama logic kept as is) ---
        logger.info(f"PanelManager calling Ollama API with prompt: {personality_prompt[:50]}...")
        url = "http://127.0.0.1:11434/api/generate" # Use 127.0.0.1
        system_prompt = "Respond in MAXIMUM 3 SENTENCES. Be casual and conversational."
        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": f"{system_prompt}\n\n{personality_prompt}\n\nQuestion/Context: {message}\n\nYour brief response (MAXIMUM 3 SENTENCES):",
            "stream": False
        }
        try:
            logger.info("PanelManager sending request to Ollama...")
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
                        return cleaned_response
                    else:
                        error_msg = f"Ollama returned status {response.status}"
                        logger.error(error_msg)
                        return f"Error: Unable to get response from Ollama (Status {response.status})"
        except asyncio.TimeoutError:
            logger.error("Exception when calling Ollama: Request timed out.")
            return "Error: Ollama request timed out."
        except Exception as e:
            error_msg = f"Exception when calling Ollama: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {str(e)}"


    async def send_agent_response(self, websocket, agent_data, prompt: str, context: str):
        # --- (send_agent_response logic kept as is, using self.* methods/attributes) ---
        agent_name = agent_data['config']['name']
        speaker_id = agent_data['index'] # Use agent's index as speaker ID
        logger.info(f"PanelManager getting response from {agent_name} for context: {context[:50]}...")
        await websocket.send_text(f"status:{agent_name} is thinking...")
        try:
            text_start_time = time.monotonic()
            response_text = await self.get_ollama_response(agent_data['config']['prompt'], prompt)
            text_end_time = time.monotonic()
            text_duration_ms = (text_end_time - text_start_time) * 1000
            if response_text and not response_text.startswith("Error:"):
                logger.info(f"Response from {agent_name}: {response_text}")
                self.history.append({ "agent": agent_name, "address": f"agent_{speaker_id}", "text": response_text, "timestamp": datetime.utcnow().isoformat() + "Z" })
                await websocket.send_text(f"message:{agent_name}: {response_text} ({text_duration_ms:.0f}ms)")
                if self.tts_available and self.tts:
                    await websocket.send_text(f"status:{agent_name} generating audio...")
                    audio_start_time = time.monotonic()
                    mp3_path = await self.tts.generate_audio_and_convert(response_text, speaker_id=speaker_id)
                    audio_end_time = time.monotonic()
                    audio_duration_ms = (audio_end_time - audio_start_time) * 1000
                    if mp3_path:
                        # Ensure static dir exists or handle path differently if needed at runtime
                        # This assumes 'static' exists relative to where app.py runs from.
                        # If running via Nix, the path needs careful consideration.
                        try:
                             web_audio_path = os.path.relpath(mp3_path, 'static')
                        except ValueError:
                             # Handle case where paths are on different drives or static doesn't exist
                             logger.error(f"Could not create relative path for audio file: {mp3_path}. Serving absolute path.")
                             web_audio_path = mp3_path # Fallback, may not work in browser

                        await websocket.send_text(f"audio:{agent_name}:{web_audio_path}:{audio_duration_ms:.0f}")
                        await websocket.send_text(f"status:{agent_name} audio ready")
                        logger.info(f"Audio generated for {agent_name}: {web_audio_path}")
                    else:
                        await websocket.send_text(f"message:System: Audio generation failed for {agent_name}.")
                        await websocket.send_text(f"status:{agent_name} ready (TTS unavailable)")
                        logger.warning(f"Audio generation failed for {agent_name}")
                else:
                     await websocket.send_text(f"status:{agent_name} ready (TTS unavailable)")
                return response_text
            else:
                logger.error(f"{agent_name} returned an error or empty response: {response_text}")
                await websocket.send_text(f"message:System: {agent_name} did not provide a valid response. ({text_duration_ms:.0f}ms)")
                await websocket.send_text(f"status:{agent_name} error")
                return None
        except Exception as e:
            logger.error(f"Error processing response from {agent_name}: {e}", exc_info=True)
            await websocket.send_text(f"message:System: Error getting response from {agent_name}")
            await websocket.send_text(f"status:{agent_name} error")
            return None

# --- Create the single PanelManager instance ---
panel_manager = PanelManager()
logger.info("Global PanelManager instance created.")

# --- Conversation Loop (Keep as refactored before) ---
async def run_conversation_loop(websocket, manager: PanelManager, max_turns=10):
    logger.info(f"Conversation loop starting for {manager.num_agents} agents.")
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
        logger.info(f"Turn {turn}: {agent_name}'s turn.")
        await websocket.send_text(f"status:Waiting for {agent_name}...")
        context_prompt = f"Current conversation context (last 3 messages):\n"
        history_context = "\n".join([f"{msg['agent']}: {msg['text']}" for msg in manager.history[-3:]])
        context_prompt += history_context + f"\n\nYour turn, {agent_name}. Respond to the last message: '{last_message_text}'"
        response_text = await manager.send_agent_response(websocket, agent_data, context_prompt, last_message_text)
        if response_text: last_message_text = response_text
        else: last_message_text = f"(Agent {agent_name} failed to respond). {agent_name}, please try again or comment on the previous message: '{last_message_text}'"
        current_agent_turn_index += 1
        await asyncio.sleep(2)
    logger.info(f"Conversation loop ended. Reason: {'Stopped by manager' if not manager.conversation_running else 'Max turns reached'}")
    if manager.active: manager.status = f"Panel active ({manager.num_agents} agents)"
    await websocket.send_text("status:Conversation Ended")
    manager.conversation_running = False

# --- WebSocket Endpoint Handler (Keep as refactored before) ---
async def websocket_handler(request):
    global panel_manager
    # --- Need aiohttp.web here! Add it back to imports ---
    from aiohttp import web # Add this import back
    from aiohttp import WSMsgType # Import WSMsgType

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("WebSocket connection established.")
    panel_manager.websocket = ws
    if panel_manager.active and not panel_manager.conversation_running:
         panel_manager.status = f"Panel active ({panel_manager.num_agents} agents)"
    await ws.send_text("status:Connected to Backend Websocket")
    await ws.send_text(f"panel_state:{json.dumps(panel_manager.get_status_data())}")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT: # Use WSMsgType.TEXT
                data = msg.data
                logger.info(f"WS Received message: {data}")
                if data.startswith("start_conversation:"):
                    if not panel_manager.active:
                        try:
                            num_agents_str = data.split(":")[1]
                            num_agents = int(num_agents_str)
                            if 1 <= num_agents <= len(AGENTS):
                                 logger.info(f"WS Received start_conversation command for {num_agents} agents.")
                                 await ws.send_text(f"message:System: Backend received start command for {num_agents} agents...")
                                 await panel_manager.start_panel(num_agents)
                            else:
                                await ws.send_text(f"message:System: Invalid number of agents ({num_agents}). Max is {len(AGENTS)}.")
                        except Exception as e:
                            logger.error(f"Error processing start_conversation via WS: {e}", exc_info=True)
                            await ws.send_text(f"message:System: Error starting conversation: {e}")
                    else:
                        await ws.send_text("message:System: Panel is already active.")
                elif data == "stop_conversation":
                    logger.info("WS Received stop_conversation command.")
                    if panel_manager.active:
                        await panel_manager.stop_panel()
                        await ws.send_text("message:System: Backend received stop command.")
                    else:
                        await ws.send_text("message:System: Panel is not currently active.")
            elif msg.type == WSMsgType.ERROR: # Use WSMsgType.ERROR
                logger.error(f'WebSocket connection closed with exception {ws.exception()}')
                break
    except Exception as e:
         logger.error(f"Error during WebSocket communication: {e}", exc_info=True)
    finally:
        logger.info('WebSocket connection closed.')
        if panel_manager.websocket == ws:
             panel_manager.websocket = None
             if panel_manager.active:
                 logger.info("Controller WebSocket disconnected, stopping panel.")
                 await panel_manager.stop_panel()
    return ws

# --- Removed __main__ block ---