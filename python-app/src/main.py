from uagents import Agent, Bureau, Context, Model
import asyncio
import aiohttp
import re
from datetime import datetime

# Define the message model for agent communication
class Message(Model):
    text: str

# Personality prompts to configure the agents
SAYLOR_PROMPT = "You are Michael Saylor, a Bitcoin maximalist. Respond in character, emphasizing your strong belief in Bitcoin's value and long-term potential."
MUSK_PROMPT = "You are Elon Musk, a tech entrepreneur with interests in crypto. Respond in character, with a mix of curiosity, innovation, and occasional humor."

# Create the agents with seeds for deterministic addresses
saylor_agent = Agent(name="Michael Saylor", seed="michaelsaylor_seed")
musk_agent = Agent(name="Elon Musk", seed="elonmusk_seed")

# Global flag to control the conversation duration
conversation_active = True

# Asynchronous function to call the local Ollama LLM API
async def get_ollama_response(personality_prompt: str, message: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:1.5b",  # Adjust the model name as needed
        "prompt": f"{personality_prompt}\n\n{message}",
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data["response"]
            else:
                return "Error: Unable to get response from Ollama"

# Function to extract conversational text (remove <think> tags)
def extract_conversation(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Function to display the message with UTC timestamp and stream words
async def display_message(agent_name: str, agent_address: str, text: str):
    # Get current UTC time
    now = datetime.utcnow()
    date_str = now.strftime("%d-%B-%Y")  # e.g., 21-March-2025
    if now.minute == 0 and now.second == 0:
        time_str = now.strftime("%I%p").lower()  # e.g., 10pm
    else:
        time_str = now.strftime("%I:%M:%S %p").lower()  # e.g., 10:01:23 pm
    
    # Print separator and timestamp with agent info
    print("*" * 120)
    print(f"{date_str} UTC time {time_str} - {agent_name} ({agent_address}):")
    
    # Stream the message word by word
    conversational_text = extract_conversation(text)
    words = conversational_text.split()
    for word in words:
        print(word, end=" ", flush=True)
        await asyncio.sleep(0.1)  # 0.1-second delay between words
    print()  # Newline after the message

# Message handler for Michael Saylor agent
@saylor_agent.on_message(model=Message)
async def handle_saylor_message(ctx: Context, sender: str, msg: Message):
    if not conversation_active:
        return
    response_text = await get_ollama_response(SAYLOR_PROMPT, msg.text)
    await display_message(ctx.agent.name, ctx.agent.address, response_text)
    await ctx.send(musk_agent.address, Message(text=response_text))

# Message handler for Elon Musk agent
@musk_agent.on_message(model=Message)
async def handle_musk_message(ctx: Context, sender: str, msg: Message):
    if not conversation_active:
        return
    response_text = await get_ollama_response(MUSK_PROMPT, msg.text)
    await display_message(ctx.agent.name, ctx.agent.address, response_text)
    await ctx.send(saylor_agent.address, Message(text=response_text))

# Startup event to initiate the conversation and schedule the stop
@saylor_agent.on_event("startup")
async def start_conversation(ctx: Context):
    initial_message = "Hey Elon, what do you think about the current Bitcoin price? I say it’s the future of money!"
    await display_message(ctx.agent.name, ctx.agent.address, initial_message)
    await ctx.send(musk_agent.address, Message(text=initial_message))
    asyncio.create_task(stop_conversation())

# Function to stop the conversation after 30 minutes
async def stop_conversation():
    await asyncio.sleep(1800)  # 30 minutes = 1800 seconds
    global conversation_active
    conversation_active = False
    print("\nConversation ended after 30 minutes.")

# Set up and run the bureau
bureau = Bureau()
bureau.add(saylor_agent)
bureau.add(musk_agent)

if __name__ == "__main__":
    bureau.run()