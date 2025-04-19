# ./python-app/src/main.py
import os
os.chdir(os.path.dirname(__file__))  # Set working directory to ./python-app/src

import logging
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiohttp
import re
from datetime import datetime
from uagents import Agent, Context, Model # Assuming uagents is still needed for Agent definition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the static/audio directory exists (optional for now)
# os.makedirs("static/audio", exist_ok=True)

app = FastAPI()

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
                    # Clean response (remove potential leading/trailing whitespace)
                    response_text = response_text.strip()
                    logger.info(f"Received response from Ollama: {response_text[:100]}...")
                    return response_text
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
    logger.info("WebSocket connection attempt.")
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted.")
        await websocket.send_text("status:Connected") # Send confirmation

        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message: {data}")

            if data.startswith("start_conversation:"):
                try:
                    num_agents_str = data.split(":")[1]
                    num_agents = int(num_agents_str)
                    logger.info(f"Received start_conversation command for {num_agents} agents.")
                    await websocket.send_text(f"message:System: Starting conversation with {num_agents} agents...")
                    await start_conversation(websocket, num_agents)
                except Exception as e:
                    logger.error(f"Error processing start_conversation: {e}")
                    await websocket.send_text(f"message:System: Error starting conversation: {e}")

            elif data == "stop_conversation":
                logger.info("Received stop_conversation command.")
                await websocket.send_text("status:Conversation stopped")
                # Add logic to actually stop agent tasks if needed

            # Add handling for other message types if necessary

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        # Try to close gracefully
        try:
            await websocket.close(code=1011) # Internal Error
        except:
            pass

# --- Conversation Logic (Keep basic structure) ---
async def start_conversation(websocket: WebSocket, num_agents: int):
    global current_agent_index, active_agents
    active_agents.clear()
    current_agent_index = 0
    logger.info(f"Initializing {num_agents} agents.")

    for i in range(min(num_agents, len(AGENTS))):
        agent_config = AGENTS[current_agent_index]
        logger.info(f"Initializing Agent: {agent_config['name']}")
        # agent = Agent(...) # Create agent instance if needed
        agent_data = {"config": agent_config, "index": current_agent_index}
        active_agents.append(agent_data)
        await websocket.send_text(f"message:System: Agent {agent_config['name']} is ready.")
        current_agent_index = (current_agent_index + 1) % len(AGENTS)

    logger.info("Agents initialized. Starting interaction.")
    await websocket.send_text("status:Conversation Ready")

    if active_agents:
        # Start with the first agent
        await get_initial_response(websocket, active_agents[0])
        # Chain responses
        if len(active_agents) > 1:
            await asyncio.sleep(2) # Delay between turns
            await get_agent_response(websocket, active_agents[1], "What are your thoughts on the initial statement?")

async def get_initial_response(websocket, agent_data):
    agent_name = agent_data['config']['name']
    logger.info(f"Getting initial response from {agent_name}")
    try:
        response = await get_ollama_response(
            agent_data['config']['prompt'],
            "Please introduce yourself briefly and share one thought about Bitcoin."
        )
        if response:
            await websocket.send_text(f"message:{agent_name}: {response}")
    except Exception as e:
        logger.error(f"Error getting initial response from {agent_name}: {e}")
        await websocket.send_text(f"message:System: Error getting response from {agent_name}")

async def get_agent_response(websocket, agent_data, prompt):
    agent_name = agent_data['config']['name']
    logger.info(f"Getting response from {agent_name} for prompt: {prompt[:50]}...")
    try:
        response = await get_ollama_response(
            agent_data['config']['prompt'],
            prompt
        )
        if response:
            await websocket.send_text(f"message:{agent_name}: {response}")
    except Exception as e:
        logger.error(f"Error getting response from {agent_name}: {e}")
        await websocket.send_text(f"message:System: Error getting response from {agent_name}")


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") # Use uvicorn's logging