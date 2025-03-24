# ./python-app/src/main.py
import os
os.chdir(os.path.dirname(__file__))  # Set working directory to ./python-app/src

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uagents import Agent, Context, Model
import asyncio
import torchaudio
import aiohttp
import re
from datetime import datetime
from sesame_tts import SesameTTS  # Assuming this is your TTS library
import torch

# Ensure the static/audio directory exists
os.makedirs("static/audio", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define a message model
class Message(Model):
    text: str

# Agent configurations
AGENTS = [
    {"name": "Michael Saylor", "seed": "michaelsaylor_seed", 
     "prompt": "You are Michael Saylor, a Bitcoin maximalist speaking on a panel. EXTREMELY IMPORTANT: Keep responses to MAXIMUM 3 SENTENCES. Be passionate about Bitcoin as digital gold."},
    {"name": "Elon Musk", "seed": "elonmusk_seed", 
     "prompt": "You are Elon Musk on a panel. EXTREMELY IMPORTANT: Keep responses to MAXIMUM 3 SENTENCES. Be quirky, mention Mars or Tesla occasionally."},
    {"name": "Vitalik Buterin", "seed": "vitalik_seed", 
     "prompt": "You are Vitalik Buterin on a panel. EXTREMELY IMPORTANT: Keep responses to MAXIMUM 3 SENTENCES. Talk about Ethereum simply."},
    {"name": "Satoshi Nakamoto", "seed": "satoshi_seed", 
     "prompt": "You are Satoshi Nakamoto on a panel. EXTREMELY IMPORTANT: Keep responses to MAXIMUM 3 SENTENCES. Be mysterious but insightful."}
]

conversation_active = False
try:
    # Use the internal models directory
    tts = SesameTTS(device="cpu", model_dir="/models")
    tts_available = True
    print("INFO:     TTS system initialized successfully")
except Exception as e:
    tts = None
    tts_available = False
    print(f"ERROR:    Failed to initialize TTS system: {str(e)}")
    print("INFO:     Running without TTS capabilities")
active_agents = []
current_agent_index = 0

# Function to extract conversational text (remove <think> tags)
def extract_conversation(text: str) -> str:
    """Remove <think> tags and their content from the text"""
    # Check if the text contains <think> tags
    if "<think>" in text:
        # Remove all content between <think> and </think> tags
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any remaining tags if the closing tag was missed
        cleaned_text = re.sub(r'<think>.*', '', cleaned_text, flags=re.DOTALL)
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
    return text

# Asynchronous function to call the local Ollama LLM API
async def get_ollama_response(personality_prompt: str, message: str) -> str:
    print(f"INFO:     Calling Ollama API with prompt: {personality_prompt[:50]}...")
    url = "http://localhost:11434/api/generate"
    
    # Use a system prompt to enforce brevity
    system_prompt = """
    VERY IMPORTANT INSTRUCTIONS:
    1. Respond in MAXIMUM 3 SENTENCES TOTAL
    2. Be casual and conversational, as if on a TV panel
    3. Use simple, direct language
    4. DO NOT use formal academic language
    5. DO NOT give long explanations
    """
    
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": f"{system_prompt}\n\n{personality_prompt}\n\nQuestion/Context: {message}\n\nYour brief response (MAXIMUM 3 SENTENCES):",
        "stream": False
    }
    
    try:
        print(f"INFO:     Sending request to Ollama...")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data["response"]
                    print(f"INFO:     Received response from Ollama: {response_text[:50]}...")
                    
                    # Force limit the response to 3 sentences
                    sentences = response_text.split('.')
                    if len(sentences) > 3:
                        limited_response = '.'.join(sentences[:3]) + '.'
                        return limited_response
                    
                    return response_text
                else:
                    error_msg = f"ERROR:    Ollama returned status {response.status}"
                    print(error_msg)
                    return f"Error: Unable to get response from Ollama (Status {response.status})"
    except Exception as e:
        error_msg = f"ERROR:    Exception when calling Ollama: {str(e)}"
        print(error_msg)
        return f"Error: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global conversation_active, current_agent_index
    await websocket.accept()
    print("INFO:     WebSocket connection accepted")
    
    # Save websocket to use in agent message handlers
    app.state.websocket = websocket
    
    # Send initial connection status to client
    await websocket.send_text("status:connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            print(f"INFO:     Received message: {data}")
            
            if data.startswith("start_conversation:"):
                num_agents = int(data.split(":")[1])
                conversation_active = True
                current_agent_index = 0
                await websocket.send_text("status:starting_conversation")
                print(f"INFO:     Starting conversation with {num_agents} agents")
                await start_conversation(websocket, num_agents)
            elif data == "stop_conversation":
                conversation_active = False
                await websocket.send_text("status:conversation_stopped")
                print("INFO:     Conversation stopped")
            elif data.startswith("message:"):
                message_text = data[len("message:"):]
                await websocket.send_text("status:processing_message")
                print(f"INFO:     Processing message: {message_text}")
                await handle_message(websocket, message_text)
    except WebSocketDisconnect:
        print("INFO:     WebSocket disconnected")
        conversation_active = False

async def generate_and_send_audio(websocket, agent_name, text, speaker_id):
    """Function to generate and send audio asynchronously"""
    try:
        print(f"INFO:     Starting TTS for {agent_name} with speaker_id={speaker_id}")
        await websocket.send_text(f"status:{agent_name}_generating_audio")
        
        audio_path = f"static/audio/{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        try:
            # Try to generate audio using our TTS instance
            print(f"INFO:     TTS generating audio for text: '{text[:50]}...' with speaker_id={speaker_id}")
            
            # Attempt to generate audio
            try:
                audio = await tts.generate_audio(text, speaker_id=speaker_id)
                print(f"INFO:     TTS completed for {agent_name}")
                
                # Save the audio file
                print(f"INFO:     Saving audio to {audio_path}")
                torchaudio.save(audio_path, audio.unsqueeze(0), tts.sample_rate)
                print(f"INFO:     Audio saved successfully")
                
                # Send the audio path to the client
                await websocket.send_text(f"audio:{audio_path}")
                print(f"INFO:     Audio path sent to client for {agent_name}")
            
            except Exception as inner_e:
                # If tensor mismatch error, create a dummy audio file
                print(f"WARNING:  TTS tensor mismatch error, creating fallback audio: {str(inner_e)}")
                
                # Create a silent audio file
                sample_rate = 24000
                silent_duration = 0.5  # half second of silence
                dummy_audio = torch.zeros(int(sample_rate * silent_duration))
                
                # Save the silent audio
                torchaudio.save(audio_path, dummy_audio.unsqueeze(0), sample_rate)
                print(f"INFO:     Fallback audio saved to {audio_path}")
                
                # Send placeholder audio path
                await websocket.send_text(f"audio:{audio_path}")
                await websocket.send_text(f"status:fallback_audio_for_{agent_name}")
                print(f"INFO:     Fallback audio sent for {agent_name}")
                
        except Exception as e:
            print(f"ERROR:    TTS failed for {agent_name}: {str(e)}")
            await websocket.send_text(f"status:audio_failed_for_{agent_name}")
            await websocket.send_text(f"error:TTS generation failed: {str(e)}")
    except Exception as e:
        print(f"ERROR:    Error in audio generation process: {str(e)}")
        await websocket.send_text(f"status:audio_process_failed_for_{agent_name}")

async def start_conversation(websocket: WebSocket, num_agents: int):
    global current_agent_index
    active_agents.clear()  # Clear previous agents
    print(f"INFO:     Starting conversation with {num_agents} agents")
    
    agent_data_list = []
    for i in range(min(num_agents, len(AGENTS))):
        agent_config = AGENTS[current_agent_index]
        print(f"INFO:     Creating agent {agent_config['name']} with seed {agent_config['seed']}")
        
        # Create the agent
        agent = Agent(name=agent_config["name"], seed=agent_config["seed"], port=None)  # Disable server
        
        # Store the agent with its configuration
        agent_data = {
            "agent": agent,
            "config": agent_config,
            "index": current_agent_index
        }
        agent_data_list.append(agent_data)
        active_agents.append(agent_data)
        
        # Notify the frontend
        await websocket.send_text(f"Agent {agent_config['name']} joined the conversation.")
        print(f"INFO:     Agent {agent_config['name']} added to active agents (Total: {len(active_agents)})")
        
        current_agent_index = (current_agent_index + 1) % len(AGENTS)
    
    print(f"INFO:     All {len(active_agents)} agents initialized and ready")
    await websocket.send_text("status:conversation_ready")
    
    # Start an initial conversation with the first agent
    if len(agent_data_list) > 0:
        await get_initial_response(websocket, agent_data_list[0])
        
        # Then get responses from other agents too
        if len(agent_data_list) > 1:
            # Small delay before next agent responds
            await asyncio.sleep(2)
            for agent_data in agent_data_list[1:]:
                await get_agent_response(websocket, agent_data, "Continue the conversation about cryptocurrency and blockchain technology. Keep your response brief and conversational, like you're speaking on a panel.")

# Function to get initial response
async def get_initial_response(websocket, agent_data):
    agent_name = agent_data['config']['name']
    print(f"INFO:     Getting initial message from {agent_name}")
    await websocket.send_text(f"status:{agent_name}_thinking")
    
    initial_prompt = "Introduce yourself briefly and start a conversation about cryptocurrency. Keep your response brief and conversational, like you're speaking on a panel."
    await websocket.send_text(f"status:generating_initial_message")
    
    try:
        response = await get_ollama_response(
            agent_data['config']['prompt'], 
            initial_prompt
        )
        
        # Clean the response and ensure it's not too long
        clean_response = extract_conversation(response)
        clean_response = shorten_response(clean_response)
        
        # Send the text immediately
        await websocket.send_text(f"message:{agent_name}: {clean_response}")
        print(f"INFO:     Sent initial message from {agent_name}")
        
        # Generate audio in the background
        asyncio.create_task(
            generate_and_send_audio(
                websocket, 
                agent_name, 
                clean_response, 
                agent_data['index'] % 2
            )
        )
    except Exception as e:
        print(f"ERROR:    Failed to get initial response from {agent_name}: {str(e)}")
        await websocket.send_text(f"status:error_getting_response_from_{agent_name}")
        await websocket.send_text(f"message:{agent_name}: Sorry, I'm having trouble connecting. Let's try again in a moment.")

# Function to get response from an agent
async def get_agent_response(websocket, agent_data, prompt):
    agent_name = agent_data['config']['name']
    print(f"INFO:     Getting response from {agent_name} to: {prompt}")
    await websocket.send_text(f"status:{agent_name}_thinking")
    
    try:
        response = await get_ollama_response(
            agent_data['config']['prompt'],
            prompt
        )
        
        # Clean and shorten the response
        clean_response = extract_conversation(response)
        clean_response = shorten_response(clean_response)
        
        # Send the text immediately
        await websocket.send_text(f"message:{agent_name}: {clean_response}")
        print(f"INFO:     Sent response from {agent_name}")
        
        # Generate audio in the background
        asyncio.create_task(
            generate_and_send_audio(
                websocket, 
                agent_name, 
                clean_response, 
                agent_data['index'] % 2
            )
        )
    except Exception as e:
        print(f"ERROR:    Failed to get response from {agent_name}: {str(e)}")
        await websocket.send_text(f"status:error_getting_response_from_{agent_name}")
        await websocket.send_text(f"message:{agent_name}: Sorry, I couldn't process that. Let me try again later.")

# Function to handle user messages
async def handle_message(websocket: WebSocket, message: str):
    if not active_agents or not conversation_active:
        print("INFO:     No active agents or conversation not active")
        return
    
    print(f"INFO:     Processing user message: '{message}'")
    await websocket.send_text(f"status:sending_to_agents")
    
    # Process responses from each agent sequentially
    for agent_data in active_agents:
        await get_agent_response(websocket, agent_data, message)
        # Add a small delay between responses
        await asyncio.sleep(1.5)
    
    await websocket.send_text("status:all_responses_sent")

# Helper function to shorten responses
def shorten_response(text):
    # If the response is too long, truncate it
    if len(text) > 500:
        # Try to find a good sentence break to truncate at
        sentences = text.split('. ')
        shortened = ""
        for sentence in sentences:
            if len(shortened) + len(sentence) + 2 < 500:  # +2 for ". "
                shortened += sentence + ". "
            else:
                break
        return shortened.strip()
    return text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)