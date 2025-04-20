# OpenxAI AI Agent Panel (Python Application)

This directory contains the Python backend and frontend for the OpenxAI AI Agent Panel application. It simulates a panel discussion featuring multiple AI agents with distinct personalities, generating both text and audio responses in real-time.

## Overview

The application uses FastAPI to serve a web interface where users can initiate a conversation. AI agents, defined by specific personality prompts, respond sequentially. Text responses are generated using a locally running Ollama instance, and corresponding audio is generated using the Sesame CSM-1B text-to-speech model. Communication between the frontend and backend happens via WebSockets.

## Features

*   **Simulated Multi-Agent Panel:** Configurable AI agents (e.g., Michael Saylor, Elon Musk) with distinct personalities participate in a discussion.
*   **Real-time Text Generation:** Leverages a local Ollama instance (specifically tested with `deepseek-r1:1.5b`) to generate conversational text for each agent.
*   **Real-time Audio Generation (TTS):** Uses the Sesame CSM-1B model to convert generated text into speech audio files (MP3).
*   **Web-Based Interface:** Simple HTML/CSS/JavaScript frontend served by FastAPI for initiating conversations and viewing/hearing responses.
*   **WebSocket Communication:** Enables real-time updates between the backend server and the user's browser.
*   **Local Model Execution:** Designed to run entirely with locally downloaded models for both LLM and TTS.

## Technology Stack

*   **Backend:** Python 3.10+, FastAPI, Uvicorn
*   **WebSockets:** `websockets` library (via `uvicorn[standard]`)
*   **Text Generation LLM:** Ollama (running `deepseek-r1:1.5b` locally)
*   **Text-to-Speech (TTS):** Sesame CSM-1B
    *   Core Model Architecture: Defined in `models.py` (utilizing `torchtune`)
    *   Text Tokenizer: Llama 3.2 1B Tokenizer (`transformers`)
    *   Audio Tokenizer: Mimi (`moshi`)
*   **Audio Handling:** `torch`, `torchaudio`, `ffmpeg` (for MP3 conversion)
*   **Async HTTP:** `aiohttp` (for calling Ollama API)
*   **Frontend:** HTML, CSS, JavaScript

## Setup & Installation

**1. Prerequisites:**

*   **Python:** Version 3.10 or higher recommended.
*   **pip:** Python package installer.
*   **Git:** For cloning the repository (if not already done).
*   **Ollama:** Must be installed and running locally. [Ollama Website](https://ollama.ai/)
*   **ffmpeg:** Required for converting generated WAV audio to MP3. Install via your system's package manager (e.g., `brew install ffmpeg` on macOS, `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu).

**2. Clone Repository:** (If you haven't already)
   ```bash
   git clone <your-repository-url>
   cd <your-repository-name>
   ```

**3. Create Virtual Environment:** (Recommended)
   Navigate to the *root* of the project directory (the parent of `python-app`) and run:
   ```bash
   python3 -m venv venv_py310 # Or your preferred venv name
   source venv_py310/bin/activate # On Windows use `venv_py310\Scripts\activate`
   ```

**4. Install Python Dependencies:**
   Install the required packages using pip:
   ```bash
   pip install fastapi "uvicorn[standard]" websockets aiohttp torch torchaudio transformers tokenizers huggingface_hub moshi torchtune packaging attrs uagents
   ```
   *(Note: `torchtune` might require specific PyTorch versions. Adjust if necessary based on compatibility.)*
   *(Note: `uagents` is imported but the core conversation logic currently uses custom async functions in `main.py`.)*

**5. Download Models:**
   This application requires several models to be downloaded manually and placed in a specific structure within a `models` directory located at the **root of the project** (i.e., sibling to the `python-app` directory).

   **Required Structure:**
   ```
   <PROJECT_ROOT>/
   ├── models/
   │   ├── csm-1b/             # <-- Sesame CSM-1B model files
   │   │   ├── pytorch_model.bin  # or model.safetensors, csm_weights.pt, etc.
   │   │   └── config.json        # and other necessary config files
   │   ├── llama-3.2-1b/       # <-- Llama 3.2 1B Tokenizer files
   │   │   ├── tokenizer.json
   │   │   ├── tokenizer_config.json
   │   │   └── ... (other tokenizer files)
   │   └── kyutai/
   │       └── mimi/             # <-- Mimi Audio Tokenizer weights
   │           └── mimi_weight.pt # or mimi_std.pt
   ├── python-app/             # <-- This application's code
   │   └── ...
   └── ... (other project files like .gitignore, README.md)
   ```
   *   You will need to obtain these models from their official sources (e.g., Hugging Face Hub). Search for `sesame/csm-1b`, `meta-llama/Llama-3.2-1B`, and the Mimi model weights (`mimi_std.pt` or `mimi_weight.pt`, often found in `collabora/whisperspeech` or similar repos).

**6. Run Ollama Model:**
   Ensure the Ollama service is running and pull the required model:
   ```bash
   ollama run deepseek-r1:1.5b
   ```
   Keep this running in a separate terminal.

## Running the Application

1.  Navigate to the `src` directory within `python-app`:
    ```bash
    cd python-app/src
    ```
2.  Start the FastAPI server using Uvicorn:
    ```bash
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload` enables auto-reloading when code changes are detected (useful for development).

## Usage

1.  Open your web browser and navigate to `http://127.0.0.1:8000`.
2.  You should see the "AI Panel Discussion" interface.
3.  The status bar will indicate the WebSocket connection status ("Connected").
4.  Select the number of agents (1-2 based on current configuration) to participate.
5.  Optionally, check/uncheck "Auto-play audio".
6.  Click "Start Conversation".
7.  The backend will initialize the agents and start the conversation loop.
8.  Text responses from agents (generated by Ollama) will appear in the "Conversation" panel.
9.  Audio generation (TTS) will start for each text response (if models loaded correctly). Status updates and timing information will appear next to the text.
10. If audio generation is successful, a play button (<i class="fas fa-play"></i>) will appear. Click it to play the audio, or let it auto-play if the checkbox is selected.
11. Audio files are saved in `/python-app/src/static/audio/`.
12. Click "Stop Conversation" to end the loop.

## File Structure (`python-app/`)
python-app/
├── src/
│ ├── main.py # FastAPI app, WebSocket endpoint, conversation loop
│ ├── models.py # CSM Model architecture definition (using torchtune)
│ ├── generator.py # Handles loading models, tokenizers (text/audio), TTS generation logic
│ ├── sesame_tts.py # Wrapper for TTS, manages audio generation & conversion process
│ ├── static/ # Frontend static files
│ │ ├── style.css
│ │ └── script.js
│ └── templates/ # HTML templates
│ └── index.html
└── README.md # This file