# ./python-app/src/main.py
import os
# Set working directory EARLY before other imports might trigger logging
# Note: This assumes main.py is run from within python-app/src
# If run from root, adjust path finding or remove os.chdir
app_dir = os.path.dirname(__file__)
os.chdir(app_dir)
print(f"INFO: Changed working directory to: {os.getcwd()}") # Confirm CWD

import logging
import logging.handlers # For file handler
from pathlib import Path
import asyncio # Keep other standard lib imports here
import time
import re
from datetime import datetime
import json
from aiohttp import web
import aiohttp_cors

# --- Logging Setup ---
# Determine project root relative to this file's location (src)
PROJECT_ROOT = Path(app_dir).parent.parent
LOG_FILE = PROJECT_ROOT / "logs.txt"

# Ensure log directory exists (optional, if you want logs in a subdir)
# LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configure root logger
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
log_level = logging.INFO # Set default level

# Create File Handler (Mode 'w' wipes the file on each run)
try:
    # Attempt to open in 'w' mode first to clear it
    with open(LOG_FILE, 'w') as f:
         f.write("--- Log Start ---\n") # Write a header
    file_handler = logging.FileHandler(LOG_FILE, mode='a') # Append after initial wipe
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level) # Log INFO and above to file
    logging.getLogger().addHandler(file_handler)
    print(f"INFO: Logging configured to file: {LOG_FILE}")
except Exception as e:
    print(f"ERROR: Failed to configure file logging to {LOG_FILE}: {e}")
    # Continue without file logging if setup fails


# Create Console Handler (to still see *some* output)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
# Set console level higher to reduce verbosity (e.g., WARNING)
# You'll still see INFO from our direct logger calls below, but less library noise
console_handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(console_handler)

# Set root logger level (acts as a filter *before* handlers)
# Set to INFO so file handler gets INFO, console handler filters further
logging.getLogger().setLevel(logging.INFO)

# --- Reduce Verbosity of Specific Libraries ---
# Set transformers logger level higher to suppress its INFO messages on console
logging.getLogger("transformers").setLevel(logging.WARNING)
# Set huggingface_hub logger level higher
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
# Set torchtune logger level higher (if it exists and is noisy)
logging.getLogger("torchtune").setLevel(logging.WARNING)
# Set datasets logger level higher
logging.getLogger("datasets").setLevel(logging.WARNING)

# Get our application-specific logger AFTER basicConfig is set
logger = logging.getLogger(__name__) # Use the name of the current module (__main__)
logger.info("Application logger initialized.") # This INFO message will go to file, but not console (by default)

# Ensure the static/audio directory exists (optional for now)
# os.makedirs("static/audio", exist_ok=True)

# --- Third-Party Imports (Move FastAPI here) ---
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect # Already imported via FastAPI, but explicit doesn't hurt
import torch
import torchaudio
import aiohttp
from uagents import Agent, Context, Model # Assuming still needed

# --- Application Imports ---
from sesame_tts import SesameTTS
# from models import ... # If needed directly in main.py
# from generator import ... # If needed directly in main.py


# --- FastAPI App Initialization ---
logger.info("Initializing FastAPI app...")
app = FastAPI() # Now FastAPI is defined

# Add CORS middleware (Important for WebSocket connections)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Agent configurations (Keep the structure)
AGENTS = [
    {"name": "Michael Saylor", "seed": "michaelsaylor_seed",
     "prompt": "You are Michael Saylor, a Bitcoin maximalist speaking on a panel. EXTREMELY IMPORTANT: Keep responses to MAXIMUM 3 SENTENCES. Be passionate about Bitcoin as digital gold."},
    {"name": "Elon Musk", "seed": "elonmusk_seed",
     "prompt": "You are Elon Musk on a panel. EXTREMELY IMPORTANT: Keep responses to MAXIMUM 3 SENTENCES. Be quirky, mention Mars or Tesla occasionally."},
    # Add other agents if needed
]

active_agents = []
current_agent_index = 0

# Add a global flag to control the conversation loop
conversation_running = False
conversation_history = []

# Initialize TTS (adjust model_dir if needed)
try:
    logger.info(f"Attempting to initialize TTS with model dir: {PROJECT_ROOT / 'models'}")
    tts = SesameTTS(device="cpu", model_dir=str(PROJECT_ROOT / "models"))
    tts_available = tts.tts_available
    logger.info(f"TTS Available: {tts_available}")
except Exception as e:
    logger.error(f"Unhandled exception during TTS init: {e}", exc_info=True) # Log full traceback to file
    tts = None
    tts_available = False

# --- Ollama Function (Keep as it was working) ---
async def get_ollama_response(personality_prompt: str, message: str) -> str:
    logger.info(f"Calling Ollama API with prompt: {personality_prompt[:50]}...")
    url = "http://localhost:11434/api/generate"
    system_prompt = "Respond in MAXIMUM 3 SENTENCES. Be casual and conversational."
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": f"{system_prompt}\n\n{personality_prompt}\n\nQuestion/Context: {message}\n\nYour brief response (MAXIMUM 3 SENTENCES):",
        "stream": False
    }
    try:
        logger.info("Sending request to Ollama...")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get("response", "Error: No response field in Ollama data")
                    logger.info(f"Raw response from Ollama: {response_text[:200]}...") # Log raw response

                    # --- Add Cleaning Logic Here ---
                    # Attempt 1: Remove <think> tags if present
                    cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

                    # Attempt 2: If no <think> tags, check if the response starts with common "thinking" phrases
                    # and try to extract the actual response part. This might need refinement based on patterns.
                    common_thinking_starts = [
                        "Okay, so", "Alright, so", "Hmm, okay,", "Thinking:",
                        "Let me break this down.", "Let me think.", "Here's my thought process:"
                    ]
                    if cleaned_response == response_text: # Only if <think> tags weren't found
                         # Find the last sentence of the thinking preamble if possible
                         sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
                         start_index = 0
                         for i, sentence in enumerate(sentences):
                             is_thinking = any(sentence.strip().startswith(phrase) for phrase in common_thinking_starts)
                             if not is_thinking:
                                 # Assume this is the start of the actual response
                                 start_index = i
                                 break
                         # Join the sentences from the assumed start of the response
                         actual_reply = " ".join(sentences[start_index:]).strip()
                         if actual_reply: # Make sure we didn't remove everything
                             cleaned_response = actual_reply

                    # Final trim and ensure it's not empty
                    cleaned_response = cleaned_response.strip()
                    if not cleaned_response:
                        logger.warning("Response cleaning resulted in empty string, returning raw response.")
                        cleaned_response = response_text.strip() # Fallback to raw if cleaning fails

                    logger.info(f"Cleaned response: {cleaned_response[:100]}...")
                    # --- End Cleaning Logic ---

                    return cleaned_response # Return the cleaned response
                else:
                    error_msg = f"Ollama returned status {response.status}"
                    logger.error(error_msg)
                    return f"Error: Unable to get response from Ollama (Status {response.status})"
    except Exception as e:
        error_msg = f"Exception when calling Ollama: {str(e)}"
        logger.error(error_msg)
        return f"Error: {str(e)}"

# --- HTML Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- WebSocket Endpoint (Simplified) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global conversation_running
    logger.info("WebSocket connection attempt.")
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted.")
        await websocket.send_text("status:Connected")

        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message: {data}")

            if data.startswith("start_conversation:"):
                if not conversation_running: # Prevent starting multiple conversations
                    try:
                        num_agents_str = data.split(":")[1]
                        num_agents = int(num_agents_str)
                        logger.info(f"Received start_conversation command for {num_agents} agents.")
                        await websocket.send_text(f"message:System: Starting conversation with {num_agents} agents...")
                        conversation_running = True # Set flag
                        asyncio.create_task(run_conversation_loop(websocket, num_agents)) # Run loop in background task
                    except Exception as e:
                        logger.error(f"Error processing start_conversation: {e}")
                        await websocket.send_text(f"message:System: Error starting conversation: {e}")
                else:
                    await websocket.send_text("message:System: Conversation already running.")

            elif data == "stop_conversation":
                logger.info("Received stop_conversation command.")
                if conversation_running:
                    conversation_running = False # Signal loop to stop
                    await websocket.send_text("message:System: Stopping conversation...")
                else:
                    await websocket.send_text("message:System: No conversation is currently running.")
            # Add handling for user messages if needed later

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
        conversation_running = False # Ensure loop stops if client disconnects
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        conversation_running = False
        try:
            await websocket.close(code=1011)
        except:
            pass

# Modify the agent response functions to include timing and audio call
async def send_agent_response(websocket: WebSocket, agent_data, prompt: str, context: str):
    agent_name = agent_data['config']['name']
    speaker_id = agent_data['index'] # Use agent's index as speaker ID

    logger.info(f"Getting response from {agent_name} for context: {context[:50]}...")
    await websocket.send_text(f"status:{agent_name} is thinking...")

    try:
        # 1. Get Text Response (and time it)
        text_start_time = time.monotonic()
        response_text = await get_ollama_response(
            agent_data['config']['prompt'],
            prompt # Pass the constructed prompt with context
        )
        text_end_time = time.monotonic()
        text_duration_ms = (text_end_time - text_start_time) * 1000

        if response_text and not response_text.startswith("Error:"):
            # Send text message with timing
            await websocket.send_text(f"message:{agent_name}: {response_text} ({text_duration_ms:.0f}ms)")

            # 2. Generate Audio (if available) (and time it)
            if tts_available and tts:
                await websocket.send_text(f"status:{agent_name} generating audio...")
                audio_start_time = time.monotonic()
                mp3_path = await tts.generate_audio_and_convert(response_text, speaker_id=speaker_id)
                audio_end_time = time.monotonic()
                audio_duration_ms = (audio_end_time - audio_start_time) * 1000

                if mp3_path:
                    # Send audio info (relative path for web access)
                    web_audio_path = os.path.relpath(mp3_path, 'static') # Make path relative to static dir
                    await websocket.send_text(f"audio:{agent_name}:{web_audio_path}:{audio_duration_ms:.0f}")
                    await websocket.send_text(f"status:{agent_name} audio ready")
                else:
                    await websocket.send_text(f"message:System: Audio generation failed for {agent_name}.")
                    await websocket.send_text(f"status:{agent_name} ready (TTS unavailable)")
            else:
                 await websocket.send_text(f"status:{agent_name} ready (TTS unavailable)")

            return response_text # Return text for conversation history

        else:
            await websocket.send_text(f"message:System: {agent_name} did not provide a valid response. ({text_duration_ms:.0f}ms)")
            await websocket.send_text(f"status:{agent_name} error")
            return None # Indicate failure

    except Exception as e:
        logger.error(f"Error getting response from {agent_name}: {e}")
        await websocket.send_text(f"message:System: Error getting response from {agent_name}")
        await websocket.send_text(f"status:{agent_name} error")
        return None

# Update the conversation loop to use the new response function
async def run_conversation_loop(websocket: WebSocket, num_agents: int, max_turns=10):
    global current_agent_index, active_agents, conversation_running, conversation_history
    active_agents.clear()
    conversation_history.clear()
    current_agent_index = 0
    logger.info(f"Initializing {num_agents} agents for conversation loop.")

    # Initialize agents
    for i in range(min(num_agents, len(AGENTS))):
        agent_config = AGENTS[current_agent_index]
        logger.info(f"Initializing Agent: {agent_config['name']}")
        agent_data = {"config": agent_config, "index": current_agent_index}
        active_agents.append(agent_data)
        await websocket.send_text(f"message:System: Agent {agent_config['name']} is ready.")
        current_agent_index = (current_agent_index + 1) % len(AGENTS)

    logger.info("Agents initialized. Starting interaction.")
    await websocket.send_text("status:Conversation Ready")

    last_message_text = "Please introduce yourself briefly and share one thought about Bitcoin."
    conversation_history.append(("System", last_message_text))

    turn = 0
    current_agent_turn_index = 0

    while conversation_running and turn < max_turns and active_agents:
        turn += 1
        agent_data = active_agents[current_agent_turn_index]
        agent_name = agent_data['config']['name']

        # Construct prompt with context
        context_prompt = f"Current conversation context (last 3 messages):\n"
        history_context = "\n".join([f"{name}: {msg}" for name, msg in conversation_history[-3:]])
        context_prompt += history_context + f"\n\nYour turn, {agent_name}. Respond to the last message: '{last_message_text}'"

        # Call the combined response function
        response_text = await send_agent_response(websocket, agent_data, context_prompt, last_message_text)

        if response_text:
            last_message_text = response_text # Update last message for next turn's context
            # Only add successful agent responses to history for context
            if not response_text.startswith("Error:"):
                 conversation_history.append((agent_name, response_text))
        else:
            # Decide how to handle failed turns (e.g., use a generic prompt)
            last_message_text = "Let's move on. What are your thoughts?"

        # Move to the next agent
        current_agent_turn_index = (current_agent_turn_index + 1) % len(active_agents)
        await asyncio.sleep(2) # Pause between turns

    logger.info(f"Conversation loop ended. Reason: {'Stopped by user' if not conversation_running else 'Max turns reached'}")
    await websocket.send_text("status:Conversation Ended")
    conversation_running = False # Ensure flag is reset

# --- Mock Data/State (Replace with your actual panel logic) ---
# In-memory store for conversation history (replace with DB or proper state management)
conversation_history = []
panel_status = "Idle"
panel_active = False
num_agents = 2 # Default or get from start request

# --- API Handlers ---

async def get_status(request):
    """ Returns the current status of the AI panel. """
    # Optional: Add secret check
    # secret = request.app['config'].get('secret')
    # if not secret or secret != request.headers.get('Authorization'):
    #     return web.json_response({"error": "invalid secret"}, status=401)

    print("GET /api/status")
    return web.json_response({
        "status": panel_status,
        "active": panel_active,
        "num_agents": num_agents,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

async def get_conversation(request):
    """ Returns the current conversation history. """
    # Optional: Add secret check
    # secret = request.app['config'].get('secret')
    # if not secret or secret != request.headers.get('Authorization'):
    #     return web.json_response({"error": "invalid secret"}, status=401)

    print(f"GET /api/conversation - returning {len(conversation_history)} messages")
    # Return the stored history
    # Add pagination/limiting in a real application
    return web.json_response({
        "history": conversation_history
        # Add other relevant state if needed
    })

async def handle_message(request):
    """ Handles incoming user messages (if your panel allows user input). """
    # Optional: Add secret check
    # secret = request.app['config'].get('secret')
    # if not secret or secret != request.headers.get('Authorization'):
    #     return web.json_response({"error": "invalid secret"}, status=401)

    if not panel_active:
        return web.json_response({"error": "Panel is not active"}, status=400)

    try:
        data = await request.json()
        message_text = data.get('message')
        user_id = data.get('userId', 'anonymous') # Get user identifier if available

        if not message_text:
            return web.json_response({"error": "Missing 'message' in request"}, status=400)

        print(f"POST /api/message - User: {user_id}, Message: {message_text}")

        # 1. Log the user message
        conversation_history.append({
            "agent": "User",
            "address": user_id,
            "text": message_text,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        # 2. TODO: Process the message with your AI panel logic
        #    - Send it to the appropriate AI agent(s).
        #    - Get the response(s).
        #    - This part is highly specific to your panel implementation.
        #    - For now, we'll just add a placeholder response.
        await asyncio.sleep(1) # Simulate processing time
        ai_response_text = f"AI Panel acknowledges: '{message_text}'"
        conversation_history.append({
            "agent": "AI Panel",
            "address": "system",
            "text": ai_response_text,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        global panel_status
        panel_status = "Processing user message..." # Update status

        return web.json_response({"status": "success", "response": ai_response_text})

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        print(f"Error handling message: {e}")
        return web.json_response({"error": "Internal server error"}, status=500)

async def handle_start(request):
    """ Starts the AI panel discussion. """
    # Optional: Add secret check
    # secret = request.app['config'].get('secret')
    # if not secret or secret != request.headers.get('Authorization'):
    #     return web.json_response({"error": "invalid secret"}, status=401)

    global panel_active, panel_status, conversation_history, num_agents
    if panel_active:
        return web.json_response({"status": "Already running"}, status=400)

    try:
        data = await request.json()
        num_agents_req = data.get('numAgents', 2)
        if not isinstance(num_agents_req, int) or not 1 <= num_agents_req <= 4:
             return web.json_response({"error": "Invalid 'numAgents' (must be 1-4)"}, status=400)
        num_agents = num_agents_req
        print(f"POST /api/start - Starting panel with {num_agents} agents")
    except Exception:
        print(f"POST /api/start - Starting panel with default {num_agents} agents (no valid body)")
        # Continue with default num_agents if body is missing/invalid

    panel_active = True
    panel_status = f"Panel starting with {num_agents} agents..."
    conversation_history = [{ # Clear history and add start message
            "agent": "System",
            "address": "system",
            "text": f"AI Panel discussion started with {num_agents} agents.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
    }]

    # TODO: Add your logic to actually start the AI agents/discussion loop
    # This might involve background tasks or triggering your agent framework.
    # For now, just update status.
    await asyncio.sleep(1) # Simulate startup
    panel_status = f"Panel active ({num_agents} agents)"

    return web.json_response({"status": "Panel started"})

async def handle_stop(request):
    """ Stops the AI panel discussion. """
    # Optional: Add secret check
    # secret = request.app['config'].get('secret')
    # if not secret or secret != request.headers.get('Authorization'):
    #     return web.json_response({"error": "invalid secret"}, status=401)

    global panel_active, panel_status
    if not panel_active:
        return web.json_response({"status": "Already stopped"}, status=400)

    print("POST /api/stop - Stopping panel")
    panel_active = False
    panel_status = "Panel stopping..."

    # TODO: Add your logic to gracefully stop the AI agents/discussion loop.

    await asyncio.sleep(0.5) # Simulate shutdown
    panel_status = "Idle"
    conversation_history.append({
            "agent": "System",
            "address": "system",
            "text": "AI Panel discussion stopped.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    return web.json_response({"status": "Panel stopped"})

# --- Add this simple test handler ---
async def handle_test(request):
    """ A very simple test endpoint. """
    print(">>> Python Backend: /api/test endpoint hit!") # Add a print statement here
    return web.json_response({
        "status": "OK",
        "message": "Python backend test endpoint is working!",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
# --- End of test handler ---

# --- Application Setup ---
async def main_app():
    app = web.Application()

    # Optional: Store config like the secret if needed
    # app['config'] = {'secret': os.environ.get('SECRET')} # Get secret from env var set by Nix

    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        # Allow the default Next.js port
        "http://localhost:3000": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*", # Allow specific headers like Authorization if needed
                allow_methods="*", # Allow GET, POST, OPTIONS etc.
            ),
        # Add your production frontend URL here if applicable
        # "https://your-frontend.com": aiohttp_cors.ResourceOptions(...)
    })

    # Define routes and apply CORS
    status_resource = cors.add(app.router.add_resource("/api/status"))
    cors.add(status_resource.add_route("GET", get_status))

    conv_resource = cors.add(app.router.add_resource("/api/conversation"))
    cors.add(conv_resource.add_route("GET", get_conversation))

    msg_resource = cors.add(app.router.add_resource("/api/message"))
    cors.add(msg_resource.add_route("POST", handle_message))

    start_resource = cors.add(app.router.add_resource("/api/start"))
    cors.add(start_resource.add_route("POST", handle_start))

    stop_resource = cors.add(app.router.add_resource("/api/stop"))
    cors.add(stop_resource.add_route("POST", handle_stop))

    # --- Add the new test route ---
    test_resource = cors.add(app.router.add_resource("/api/test"))
    cors.add(test_resource.add_route("GET", handle_test))
    # --- End of adding test route ---

    # --- REMOVE OLD ROUTES ---
    # Make sure any routes serving index.html or static files are removed
    # e.g., remove lines like:
    # app.router.add_static('/static/', ...)
    # app.router.add_get('/', handle_index)

    return app

# If running directly (without Nix wrapper/systemd), you might need this:
if __name__ == "__main__":
    # Get port from environment variable or default
    port = int(os.environ.get('PORT', 8000))
    app = asyncio.run(main_app())
    web.run_app(app, host='0.0.0.0', port=port)

# Note: If Nix/systemd manages the execution, the if __name__ == "__main__":
# block might not be strictly necessary, as the Nix setup likely calls a
# specific function or uses a runner like Gunicorn/Uvicorn/Hypercorn.
# Ensure your nix/python-app.nix correctly points to this app.