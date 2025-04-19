document.addEventListener('DOMContentLoaded', () => {
    const statusDiv = document.getElementById('status');
    const conversationDiv = document.getElementById('conversation');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const numAgentsInput = document.getElementById('numAgents');
    const autoPlayCheckbox = document.getElementById('autoPlayAudio');
    let ws = null;
    let messageIdCounter = 0; // For unique IDs

    function updateStatus(message) {
        console.log("Status:", message);
        statusDiv.textContent = message;
    }

    function addMessage(speaker, text, textTimeMs, messageId) {
        console.log("Adding message:", speaker, text);
        const msgElement = document.createElement('div');
        msgElement.className = 'message-container';
        msgElement.id = `msg-${messageId}`; // Assign unique ID

        let timeInfo = textTimeMs ? ` <span class="time-info">(Text: ${textTimeMs}ms)</span>` : '';

        msgElement.innerHTML = `
            <p class="message-text"><strong>${speaker}:</strong> ${text}${timeInfo}</p>
            <div class="audio-controls" id="audio-controls-${messageId}">
                <span class="audio-status"></span>
                <span class="audio-time"></span>
                <button class="play-button" style="display: none;" title="Play Audio">
                    <i class="fas fa-play"></i>
                </button>
            </div>
        `;
        conversationDiv.appendChild(msgElement);
        conversationDiv.scrollTop = conversationDiv.scrollHeight; // Auto-scroll
    }

    function updateAudioStatus(messageId, status, audioTimeMs = null, audioPath = null) {
        const audioControls = document.getElementById(`audio-controls-${messageId}`);
        if (!audioControls) return;

        const statusSpan = audioControls.querySelector('.audio-status');
        const timeSpan = audioControls.querySelector('.audio-time');
        const playButton = audioControls.querySelector('.play-button');

        statusSpan.textContent = `Audio: ${status}`;
        playButton.style.display = 'none'; // Hide button by default

        if (audioTimeMs !== null) {
            timeSpan.textContent = `(Gen: ${audioTimeMs}ms)`;
        } else {
            timeSpan.textContent = '';
        }

        if (status === 'Ready' && audioPath) {
            playButton.style.display = 'inline-block';
            playButton.dataset.audioSrc = `/static/${audioPath}`; // Store path relative to static
            playButton.onclick = (e) => {
                const btn = e.currentTarget;
                const audioSrc = btn.dataset.audioSrc;
                console.log(`Playing audio: ${audioSrc}`);
                const audio = new Audio(audioSrc);
                 // Handle potential play errors
                 audio.play().catch(error => console.error("Audio play failed:", error));
            };

            // Auto-play logic
            if (autoPlayCheckbox.checked) {
                 console.log(`Auto-playing audio: /static/${audioPath}`);
                 const audio = new Audio(`/static/${audioPath}`);
                 audio.play().catch(error => console.error("Auto-play failed:", error));
            }
        }
    }

    function connectWebSocket() {
        updateStatus('Connecting...');
        ws = new WebSocket(`ws://${window.location.host}/ws`); // Use relative host
    
    ws.onopen = () => {
            updateStatus('Connected');
            addMessage('System', 'WebSocket Connection Opened.', null, messageIdCounter++);
            startButton.disabled = false;
        };

        ws.onclose = () => {
            updateStatus('Disconnected. Please refresh.');
            addMessage('System', 'WebSocket Connection Closed.', null, messageIdCounter++);
            startButton.disabled = true;
            stopButton.style.display = 'none';
            startButton.style.display = 'inline-block';
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            updateStatus('Connection Error. Please refresh.');
            addMessage('System', 'WebSocket Error.', null, messageIdCounter++);
            startButton.disabled = true;
        };

    ws.onmessage = (event) => {
            const message = event.data;
            console.log('Message from server:', message);

            if (message.startsWith('status:')) {
                updateStatus(message.substring(7));
                // Extract thinking status for specific agent
                 const thinkingMatch = message.match(/status:(.*?) is thinking.../);
                 if (thinkingMatch) {
                     const agentName = thinkingMatch[1];
                     // Find the latest message from this agent and update its audio status?
                     // Or maybe just update the main status bar.
                 }
                 // Extract generating audio status
                 const generatingMatch = message.match(/status:(.*?) generating audio.../);
                 if(generatingMatch){
                    // Find latest message ID for this agent and update status
                    const agentName = generatingMatch[1];
                    const agentMessages = conversationDiv.querySelectorAll(`.message-container strong:contains("${agentName}")`);
                    if(agentMessages.length > 0){
                        const latestMessageContainer = agentMessages[agentMessages.length-1].closest('.message-container');
                        if(latestMessageContainer){
                            updateAudioStatus(latestMessageContainer.id.split('-')[1], 'Generating...');
                        }
                    }
                 }
            } else if (message.startsWith('message:')) {
                const content = message.substring(8);
                const parts = content.split(': ');
                const speaker = parts[0];
                let text = parts.slice(1).join(': ');
                let textTimeMs = null;

                // Check for timing info at the end
                const timeMatch = text.match(/\((\d+)ms\)$/);
                if (timeMatch) {
                    textTimeMs = parseInt(timeMatch[1], 10);
                    text = text.replace(/\s*\(\d+ms\)$/, '').trim(); // Remove time from text
                }

                messageIdCounter++;
                addMessage(speaker, text, textTimeMs, messageIdCounter);

            } else if (message.startsWith('audio:')) {
                // Format: audio:AgentName:path/to/audio.mp3:durationMs
                const parts = message.split(':');
                if (parts.length === 4) {
                    const agentName = parts[1];
                    const audioPath = parts[2];
                    const audioTimeMs = parseInt(parts[3], 10);

                    // Find the latest message container for this agent
                     const agentMessages = Array.from(conversationDiv.querySelectorAll('.message-container'));
                     let targetMessageId = null;
                     for (let i = agentMessages.length - 1; i >= 0; i--) {
                         const speakerStrong = agentMessages[i].querySelector('strong');
                         if (speakerStrong && speakerStrong.textContent === agentName + ":") {
                             targetMessageId = agentMessages[i].id.split('-')[1];
                             break;
                         }
                     }

                    if (targetMessageId) {
                        updateAudioStatus(targetMessageId, 'Ready', audioTimeMs, audioPath);
                    } else {
                         console.warn(`Could not find message for agent ${agentName} to attach audio.`);
                    }
                }
            } else {
                addMessage("System", message, null, ++messageIdCounter); // Handle unexpected messages
            }
        };
    }

    startButton.onclick = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const numAgents = numAgentsInput.value;
            addMessage('System', `Sending start command for ${numAgents} agents...`, null, messageIdCounter++);
            ws.send(`start_conversation:${numAgents}`);
            startButton.style.display = 'none';
            stopButton.style.display = 'inline-block';
            conversationDiv.innerHTML = ''; // Clear previous conversation
        } else {
            updateStatus('Not connected. Attempting to reconnect...');
            connectWebSocket(); // Try to reconnect if not open
        }
    };

    stopButton.onclick = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            addMessage('System', 'Sending stop command...', null, messageIdCounter++);
            ws.send('stop_conversation');
            startButton.style.display = 'inline-block';
            stopButton.style.display = 'none';
        }
    };

    // Initial connection attempt
    startButton.disabled = true; // Disable until connected
    connectWebSocket();
});