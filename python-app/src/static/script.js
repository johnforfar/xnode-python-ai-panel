document.addEventListener('DOMContentLoaded', () => {
    const statusDiv = document.getElementById('status');
    const conversationDiv = document.getElementById('conversation');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const numAgentsInput = document.getElementById('numAgents');
let ws = null;

    function updateStatus(message) {
        console.log("Status:", message);
        statusDiv.textContent = message;
    }

    function addMessage(message) {
        console.log("Adding message:", message);
        const msgElement = document.createElement('p');
        // Basic formatting for speaker: message
        if (message.includes(': ')) {
            const parts = message.split(': ');
            const speaker = parts[0];
            const text = parts.slice(1).join(': ');
            msgElement.innerHTML = `<strong>${speaker}:</strong> ${text}`;
        } else {
            msgElement.textContent = message; // For system messages
        }
        conversationDiv.appendChild(msgElement);
        conversationDiv.scrollTop = conversationDiv.scrollHeight; // Auto-scroll
    }

    function connectWebSocket() {
        updateStatus('Connecting...');
        ws = new WebSocket(`ws://${window.location.host}/ws`); // Use relative host

        ws.onopen = () => {
            updateStatus('Connected');
            addMessage('System: WebSocket Connection Opened.');
            startButton.disabled = false;
        };

        ws.onclose = () => {
            updateStatus('Disconnected. Please refresh.');
            addMessage('System: WebSocket Connection Closed.');
            startButton.disabled = true;
            stopButton.style.display = 'none';
            startButton.style.display = 'inline-block';
        };

        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            updateStatus('Connection Error. Please refresh.');
            addMessage('System: WebSocket Error.');
            startButton.disabled = true;
        };

        ws.onmessage = (event) => {
            const message = event.data;
            console.log('Message from server:', message);
            if (message.startsWith('status:')) {
                updateStatus(message.substring(7));
            } else if (message.startsWith('message:')) {
                addMessage(message.substring(8));
            } else {
                addMessage(message); // Handle unexpected messages
            }
        };
    }

    startButton.onclick = () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const numAgents = numAgentsInput.value;
            addMessage(`System: Sending start command for ${numAgents} agents...`);
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
            addMessage('System: Sending stop command...');
            ws.send('stop_conversation');
            startButton.style.display = 'inline-block';
            stopButton.style.display = 'none';
        }
    };

    // Initial connection attempt
    startButton.disabled = true; // Disable until connected
    connectWebSocket();
});