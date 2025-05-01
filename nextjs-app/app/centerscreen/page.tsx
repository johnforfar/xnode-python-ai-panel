// ./app/centerscreen/page.tsx
"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";

// Define speaker data mapping - MUST MATCH BACKEND (e.g., python-app/src/replay.py speaker_id map)
const speakerIdToName: { [key: number]: string } = {
  0: "Kxi",  // Moderator
  1: "Liq",  // Inspired by Michael Saylor
  2: "Kai",  // Inspired by Peter Schiff
  3: "Vivi", // Inspired by Satoshi Nakamoto
  4: "Nn"    // Inspired by Donald Trump
};

// Define message types expected from WebSocket
type WsConnectionStatus =
  | "connecting"
  | "open"
  | "closing"
  | "closed"
  | "error";

type HistoryMessage = {
  agent: string;
  text: string;
  timestamp: string;
  // Include other fields if needed for potential future use
  address?: string;
  audioStatus?: string;
  audioUrl?: string;
};

type WebSocketMessage = {
  type: string;
  payload: any; // Adjust based on actual message structure
};

export default function CenterScreenPage() {
  const [currentText, setCurrentText] = useState<string>(
    "Waiting for conversation..."
  );
  const [currentSpeakerName, setCurrentSpeakerName] = useState<string | null>(null); // Track speaker
  const [historyLog, setHistoryLog] = useState<HistoryMessage[]>([]); // Store history
  const [wsStatus, setWsStatus] = useState<WsConnectionStatus>("closed");
  const ws = useRef<WebSocket | null>(null);

  // WebSocket Connection Effect
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    // Assuming the main WebSocket endpoint is running on the same host, adjust port if needed
    const wsUrl = `${protocol}//${window.location.hostname}:${
      window.location.protocol === "https:" ? "443" : "8000" // Default WS port
    }/ws`;

    console.log(`CenterScreen: Attempting WebSocket connection to: ${wsUrl}`);
    setWsStatus("connecting");
    const socket = new WebSocket(wsUrl);
    ws.current = socket;

    socket.onopen = () => {
      console.log(`CenterScreen: WebSocket connection established`);
      setWsStatus("open");
       // Subscribe to all speakers - necessary if backend filters non-subscribed audio
       // Even though we don't play audio here, receiving the 'audio' message
       // for the speaker is our trigger to display their text.
      Object.keys(speakerIdToName).forEach(id => {
          socket.send(btoa(JSON.stringify({ type: "subscribe", payload: parseInt(id) })));
          console.log(`CenterScreen: Subscribed to speaker ${id}`);
      });
    };

    socket.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(atob(event.data));

        if (
          message.type === "conversation_history" &&
          message.payload?.history &&
          Array.isArray(message.payload.history)
        ) {
          const newHistory: HistoryMessage[] = message.payload.history;
          setHistoryLog(newHistory); // Update the stored history log

          // Update text based on latest history entry initially, only if no speaker is active
          if (!currentSpeakerName) {
              const latestMessage = newHistory
                .slice()
                .reverse()
                .find((msg) => msg.agent !== "System");

              if (latestMessage && latestMessage.text) {
                 if (latestMessage.text !== currentText) {
                     setCurrentText(latestMessage.text);
                     // Don't set currentSpeakerName from history directly, wait for audio trigger
                 }
              } else if (newHistory.length <= 1) {
                setCurrentText("Waiting for conversation...");
                setCurrentSpeakerName(null);
              }
          }

        } else if (message.type === "audio") {
            // Update text based on the speaker whose audio chunk arrived
            const speakerId: number | undefined = message.payload?.speaker;

            if (speakerId !== undefined && historyLog.length > 0) {
                const speakerName = speakerIdToName[speakerId];
                if (speakerName) {
                    // Find the latest message from this specific speaker in our stored history
                    const speakerLastMessage = historyLog
                        .slice()
                        .reverse()
                        .find(msg => msg.agent === speakerName);

                    if (speakerLastMessage && speakerLastMessage.text) {
                         // Update text only if it's different or speaker changes
                         if (speakerLastMessage.text !== currentText || speakerName !== currentSpeakerName) {
                            setCurrentText(speakerLastMessage.text);
                            setCurrentSpeakerName(speakerName); // Track current speaker
                            console.log(`CenterScreen: Displaying text for speaker: ${speakerName}`);
                         }
                    }
                } else {
                     console.warn(`CenterScreen: Received audio chunk for unknown speaker ID: ${speakerId}`);
                }
            }

        } else if (message.type === "status_update") {
           // Optional: Handle panel stop status
           if(message.payload?.active === false && wsStatus !== 'closed') {
                setCurrentText("Panel ended.");
                setCurrentSpeakerName(null);
           }
        }
      } catch (e) {
        console.error(
          `CenterScreen: Failed to parse WebSocket message:`,
          event.data,
          e
        );
      }
    };

    socket.onerror = (event) => {
      console.error(`CenterScreen: WebSocket error:`, event);
      setWsStatus("error");
      setCurrentText("WebSocket connection error.");
      setCurrentSpeakerName(null);
    };

    socket.onclose = (event) => {
      console.log(
        `CenterScreen: WebSocket connection closed. Code: ${event.code}`
      );
      setWsStatus("closed");
      ws.current = null;
       if (event.code !== 1000) { // Don't show error on clean close
           setCurrentText("WebSocket disconnected.");
       } else {
            setCurrentText("Panel ended."); // Assume clean close means panel ended
       }
       setCurrentSpeakerName(null);
       setHistoryLog([]); // Clear history on close
    };

    // Cleanup function
    return () => {
      console.log(`CenterScreen: Cleaning up WebSocket connection.`);
      if (ws.current) {
        ws.current.close(1000, "Client navigating away"); // Send clean close code
        ws.current = null;
      }
      setWsStatus("closed");
    };
  }, []); // Empty dependency array means this runs once on mount

  return (
    // Using a key on main to potentially force re-render on critical state changes if needed,
    // but key on <p> tag is usually sufficient for text animation.
    <main className="relative min-h-screen w-full font-sans overflow-hidden text-white">
      {/* Background Image */}
      <Image
        src="/1.jpg" // Assuming 1.jpg is in the public folder
        alt="Background"
        fill
        style={{ objectFit: "cover" }}
        priority // Load the background image early
        unoptimized // Useful if the image is served locally
        className="-z-10" // Place the image behind other content
      />

      {/* Optional: Dark Scrim for Text Readability */}
      <div className="absolute inset-0 bg-black/40 -z-10"></div>

      {/* Status Indicator (Optional but helpful for debugging) */}
      <div className="absolute top-3 right-3 text-xs p-1.5 rounded bg-black/50 backdrop-blur-sm z-20 flex flex-col items-end gap-1">
         <span>WS: {wsStatus}</span>
         {currentSpeakerName && <span>Speaker: {currentSpeakerName}</span>}
      </div>

      {/* Content Container - Centering the text */}
      <div className="relative z-10 flex flex-col items-center justify-center h-screen p-6 text-center">
        {/* Use a key based on the text to force re-render for animation */}
        <p
          key={currentText} // Force re-render on text change for animation trigger
          className="text-4xl md:text-6xl lg:text-7xl font-semibold animate-fade-in" // Added simple fade-in
          style={{ textShadow: "2px 2px 8px rgba(0,0,0,0.8)" }} // Slightly stronger shadow
        >
          {currentText}
        </p>
      </div>

      {/* Basic Fade-in Animation Definition (add to your global CSS if preferred) */}
      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.5s ease-out forwards;
        }
        /* Ensure specificity if Tailwind overrides */
        main > div > p.animate-fade-in {
           animation: fadeIn 0.5s ease-out forwards !important;
        }
      `}</style>
    </main>
  );
}