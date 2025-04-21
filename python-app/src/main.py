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
import aiohttp

# --- UAGENTS Imports ---
from uagents import Agent, Bureau, Context, Model
from uagents.setup import fund_agent_if_low

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
logger.info("Application logger initialized (main.py).")

# --- Third-Party Imports (Removed FastAPI/Starlette) ---
import torch
import torchaudio
# from uagents import Agent, Context, Model # Keep if needed by panel logic, remove otherwise

# --- Application Imports ---
from sesame_tts import SesameTTS

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
"""
    },
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
"""
    },
]

# --- UAGENTS Definitions ---

# Message model for agent communication
class Message(Model):
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


# --- PanelManager Class Definition (Modified for uAgents) ---
class PanelManager:
    def __init__(self):
        logger.info("Initializing PanelManager...")
        self.status = "Idle"
        self.active = False
        # Removed conversation_running flag - uagents loop controls itself via conversation_active
        self.history = []
        self.num_agents = 0
        self.active_agents_list = [] # Store active agent instances
        self.websockets = set()
        self.bureau_task = None # Task handle for bureau.run()
        self.bureau = None      # Bureau instance

        # Removed TTS for simplicity, can be added back to handle_agent_response
        self.tts = None
        self.tts_available = False

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

    async def handle_agent_response(self, agent_name: str, agent_address: str, text: str):
        """Called by agent handlers to update history and broadcast."""
        logger.info(f"PanelManager handling response from {agent_name}: {text[:60]}...")
        if not self.active: return # Don't process if panel stopped

        message_payload = {
            "agent": agent_name, "address": agent_address,
            "text": text, "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        self.history.append(message_payload)
        await self.broadcast_message({"type": "agent_message", "payload": message_payload})
        # Optionally add TTS call here
        await self.broadcast_message({ # Reset status after message broadcast
            "type": "status_update",
            "payload": {"status": f"Panel active ({self.num_agents} agents)", "active": True, "num_agents": self.num_agents}
        })

    async def start_panel(self, num_agents_req: int):
        """Starts the uAgents Bureau and conversation."""
        global conversation_active
        if self.active: return False # Already running

        logger.info(f"Starting panel with {num_agents_req} agents using uAgents Bureau.")
        self.num_agents = num_agents_req
        self.active = True
        conversation_active = True # Enable agent message handling
        self.status = "Starting uAgents Bureau..."
        self.history = [{
            "agent": "System", "address": "system",
            "text": f"AI Panel discussion started with {self.num_agents} agents (uAgents). Topic: Bitcoin vs Gold.",
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
        await self.broadcast_message({"type": "system_message", "payload": {"text": f"uAgents panel started. {self.active_agents_list[0].name} will begin."}})

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

    async def get_ollama_response(self, personality_prompt: str, message: str) -> str:
        logger.info(f"--- (1) ENTERING get_ollama_response for prompt: {personality_prompt[:30]}... ---") # Log Entry
        url = "http://127.0.0.1:11434/api/generate"
        system_prompt = "Respond in MAXIMUM 3 SENTENCES. Be casual and conversational."
        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": f"{system_prompt}\n\n{personality_prompt}\n\nQuestion/Context: {message}\n\nYour brief response (MAXIMUM 3 SENTENCES):",
            "stream": False
        }
        try:
            logger.info(f"--- (2) Preparing to send POST to Ollama at {url} ---") # Log Before Request
            async with aiohttp.ClientSession() as session:
                logger.info(f"--- (3) Sending POST to Ollama NOW ---") # Log Immediately Before Request
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    logger.info(f"--- (4) Received Ollama status: {response.status} ---") # Log After Response
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
                        logger.info(f"--- (5) EXITING get_ollama_response (Success) ---")
                        return cleaned_response
                    else:
                        error_msg = f"Ollama returned status {response.status}"
                        logger.error(error_msg)
                        logger.info(f"--- (5) EXITING get_ollama_response (Ollama Error Status {response.status}) ---")
                        return f"Error: Unable to get response from Ollama (Status {response.status})"
        except asyncio.TimeoutError:
             logger.error("--- (E1) EXITING get_ollama_response (Timeout Error) ---")
             return "Error: Ollama request timed out."
        except Exception as e:
             logger.error(f"--- (E2) EXITING get_ollama_response (Exception: {e}) ---", exc_info=True)
             return f"Error: {str(e)}"


# --- Create the single PanelManager instance ---
panel_manager = PanelManager()
logger.info("Global PanelManager instance created.")


# --- UAGENTS Agent Handlers ---

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
    response_text = await panel_manager.get_ollama_response(SAYLOR_PROMPT, msg.text)
    cleaned_response = extract_conversation(response_text)

    logger.info("Saylor Agent: Calling PanelManager.handle_agent_response...")
    await panel_manager.handle_agent_response(ctx.agent.name, ctx.agent.address, cleaned_response)

    logger.info(f"Saylor Agent: Sending response to Schiff Agent ({schiff_agent.address}).")
    await ctx.send(schiff_agent.address, Message(text=cleaned_response))

@schiff_agent.on_message(model=Message, replies=Message)
async def handle_schiff_message(ctx: Context, sender: str, msg: Message):
    logger.info(f"Schiff Agent received message from {sender}: '{msg.text[:50]}...'")
    if not conversation_active:
        logger.info("Schiff Agent: Conversation inactive, skipping processing.")
        return

    logger.info("Schiff Agent: Calling PanelManager.get_ollama_response...")
    response_text = await panel_manager.get_ollama_response(SCHIFF_PROMPT, msg.text)
    cleaned_response = extract_conversation(response_text)

    logger.info("Schiff Agent: Calling PanelManager.handle_agent_response...")
    await panel_manager.handle_agent_response(ctx.agent.name, ctx.agent.address, cleaned_response)

    logger.info(f"Schiff Agent: Sending response to Saylor Agent ({saylor_agent.address}).")
    await ctx.send(saylor_agent.address, Message(text=cleaned_response))

# Startup event on Saylor agent to kick off the conversation
@saylor_agent.on_event("startup")
async def agent_startup(ctx: Context):
    # *** FIX HERE: Use ctx.agent.name (and ensure ctx.agent.address exists) ***
    agent_name = ctx.agent.name  # Get name from agent object
    agent_address = ctx.agent.address # Get address from agent object
    logger.info(f"{agent_name} ({agent_address}) startup event.")
    if panel_manager.active:
        initial_topic = "Let's debate the true store of value: Bitcoin versus Gold. Peter, gold has history, but isn't Bitcoin superior digital scarcity?"
        logger.info(f"{agent_name}: Sending initial message to Schiff Agent: {initial_topic}")
        # Use the retrieved agent_name and agent_address
        await panel_manager.handle_agent_response(agent_name, agent_address, initial_topic)
        await ctx.send(schiff_agent.address, Message(text=initial_topic))
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