#!/usr/bin/env python3
"""
Web UI for the AI Panel Discussion with CPU-based text-to-speech
"""
import os
import asyncio
import logging
import uuid
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Force CPU usage for all operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["NO_TORCH_COMPILE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Import our generator
from generator import load_csm_1b_local, Segment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Panel Discussion")

# Get the current directory
current_dir = Path(__file__).parent
static_dir = current_dir / "static"
templates_dir = current_dir / "templates"
audio_dir = Path("audio_outputs")
audio_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=str(templates_dir))

# Dictionary to store active connections
connections = {}

# Dictionary to store conversation history by session ID
conversations = {}

# TTS Generator (load on first use)
generator = None

# Agent data
AGENT_PERSONAS = {
    "Moderator": {
        "name": "Moderator",
        "description": "A neutral facilitator who ensures the discussion stays on track",
        "speaker_id": 0
    },
    "Expert": {
        "name": "Expert",
        "description": "A knowledgeable specialist with deep domain expertise",
        "speaker_id": 1
    },
    "Critic": {
        "name": "Critic",
        "description": "A skeptical analyst who challenges assumptions",
        "speaker_id": 2
    },
    "Innovator": {
        "name": "Innovator",
        "description": "A creative thinker who proposes new ideas and solutions",
        "speaker_id": 3
    }
}

def initialize_generator():
    """Initialize the TTS generator if not already loaded"""
    global generator
    if generator is None:
        logger.info("Loading CSM model on CPU")
        try:
            start_time = time.time()
            generator = load_csm_1b_local(device="cpu")
            logger.info(f"CSM model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load CSM model: {e}")
            raise
    return generator

async def generate_speech(text: str, speaker_id: int, output_path: str) -> str:
    """Generate speech audio for the given text and save as MP3"""
    try:
        # Initialize generator if needed
        gen = initialize_generator()
        
        # Generate audio with direct MP3 output
        logger.info(f"Generating speech for speaker {speaker_id}: '{text[:50]}...'")
        start_time = time.time()
        
        # Use empty context for now
        audio = gen.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=20000,  # 20 seconds max
            temperature=0.8,
            topk=50,
            output_mp3=True,
            output_path=output_path
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Speech generated in {generation_time:.2f} seconds")
        
        # Return MP3 path
        return f"{output_path}.mp3"
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        # Return None to indicate failure
        return None

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files"""
    file_path = audio_dir / filename
    return FileResponse(str(file_path))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    connections[session_id] = websocket
    conversations[session_id] = {
        "agents": [],
        "messages": [],
        "context": []
    }
    
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message: {data}")
            
            if data.startswith("start_conversation:"):
                num_agents = int(data.split(":")[1])
                await handle_start_conversation(session_id, num_agents)
            elif data.startswith("message:"):
                user_message = data[len("message:"):]
                await handle_user_message(session_id, user_message)
            else:
                await websocket.send_text(f"error:Unknown command: {data}")
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
        if session_id in connections:
            del connections[session_id]
        if session_id in conversations:
            del conversations[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(f"error:Internal server error: {str(e)}")
        except:
            pass

async def handle_start_conversation(session_id: str, num_agents: int):
    """Start a new conversation with the specified number of agents"""
    if session_id not in connections:
        return
    
    websocket = connections[session_id]
    await websocket.send_text("status:starting_conversation")
    
    # Select agents
    agents = list(AGENT_PERSONAS.values())[:num_agents]
    conversations[session_id]["agents"] = agents
    
    # Announce agents joining
    for agent in agents:
        await websocket.send_text(f"Agent {agent['name']} joined the conversation")
        await asyncio.sleep(0.5)  # Small delay between announcements
    
    await websocket.send_text("status:conversation_ready")
    
    # Have the moderator start the conversation
    if agents and agents[0]["name"] == "Moderator":
        await generate_agent_response(
            session_id=session_id,
            agent=agents[0],
            message="Welcome to our panel discussion. Let's begin by introducing ourselves and our perspectives."
        )

async def handle_user_message(session_id: str, message: str):
    """Handle a user message and generate agent responses"""
    if session_id not in connections or session_id not in conversations:
        return
    
    websocket = connections[session_id]
    conversation = conversations[session_id]
    agents = conversation["agents"]
    
    if not agents:
        await websocket.send_text("error:No agents in conversation")
        return
    
    await websocket.send_text("status:processing_message")
    
    # For simplicity, have the first agent respond to the user
    await generate_agent_response(
        session_id=session_id,
        agent=agents[0],
        message=f"I received your message: {message}. Let me address that."
    )
    
    # If there are multiple agents, have another agent respond
    if len(agents) > 1:
        await generate_agent_response(
            session_id=session_id,
            agent=agents[1],
            message=f"I'd like to add my perspective on that."
        )

async def generate_agent_response(session_id: str, agent: Dict[str, Any], message: str):
    """Generate a response from an agent including speech synthesis"""
    if session_id not in connections:
        return
    
    websocket = connections[session_id]
    conversation = conversations[session_id]
    
    # Send the message text to the client
    await websocket.send_text(f"message:{agent['name']}: {message}")
    
    # Generate unique filename for the audio
    filename_base = f"{agent['name']}_{uuid.uuid4().hex[:8]}"
    output_path = str(audio_dir / filename_base)
    
    # Generate the speech audio
    audio_path = await generate_speech(
        text=message,
        speaker_id=agent['speaker_id'],
        output_path=output_path
    )
    
    if audio_path:
        # Send the audio path to the client
        await websocket.send_text(f"audio:{audio_path}")
    else:
        # If generation failed, notify the client
        await websocket.send_text(f"status:fallback_audio_for_{agent['name']}")
    
    # Add to conversation history
    conversation["messages"].append({
        "agent": agent["name"],
        "message": message
    })

@app.on_event("startup")
async def startup_event():
    """Run startup tasks"""
    logger.info("Starting AI Panel Discussion web UI")
    
    # Make sure the audio directory exists
    audio_dir.mkdir(exist_ok=True)
    
    # Pre-load the generator in a separate task to avoid blocking startup
    asyncio.create_task(preload_generator())

async def preload_generator():
    """Preload the generator in the background"""
    try:
        # Add a small delay to ensure the server has started
        await asyncio.sleep(1)
        initialize_generator()
    except Exception as e:
        logger.error(f"Failed to preload generator: {e}")

if __name__ == "__main__":
    uvicorn.run("web_ui:app", host="0.0.0.0", port=8000, reload=True) 