// ./app/centerscreen/page.tsx
"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";

// Define speaker data mapping - MUST MATCH BACKEND (e.g., python-app/src/replay.py speaker_id map)
const speakerIdToName: { [key: number]: string } = {
  0: "CryptoKitty", // Moderator
  1: "MrLightning", // Inspired by Michael Saylor
  2: "PeterGoldBug", // Inspired by Peter Schiff
  3: "RealSatoshi", // Inspired by Satoshi Nakamoto
  4: "TheDon", // Inspired by Donald Trump
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
  const [historyLog, setHistoryLog] = useState<HistoryMessage[]>([]); // Store history
  const [currentSpeaker, setCurrentSpeaker] = useState<string>("");
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
      setInterval(() => {
        // Keep alive heartbeat
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(btoa(JSON.stringify({ type: "ping" })));
        }
      }, 1000);
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
    };

    socket.onclose = (event) => {
      console.log(
        `CenterScreen: WebSocket connection closed. Code: ${event.code}`
      );
      setWsStatus("closed");
      ws.current = null;
      if (event.code !== 1000) {
        // Don't show specific disconnect message on abnormal close
        console.warn(
          `WebSocket closed abnormally (code: ${event.code}), setting text to 'Panel ended'.`
        );
      }
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

  useEffect(() => {
    // Find latest message after current timestamp
    const timer = setInterval(() => {
      const latestIndex = historyLog.findIndex(
        (m) => new Date(m.timestamp).getTime() > new Date().getTime()
      );
      if (latestIndex > 0) {
        const latestItem = historyLog.at(latestIndex - 1);
        setCurrentText(latestItem?.text ?? "");
        setCurrentSpeaker(latestItem?.agent ?? "");
      }
    }, 10);

    return () => clearInterval(timer);
  }, [historyLog]);

  return (
    // Using a key on main to potentially force re-render on critical state changes if needed,
    // but key on <p> tag is usually sufficient for text animation.
    <main
      className={`relative min-h-screen w-full overflow-hidden text-white centerscreen-background ${
        currentText ? "speaking" : "" // Use 'speaking' class when someone is talking
      }`}
    >
      {/* Background Image - REMOVED */}
      {/*
      <Image
        src="/1.jpg" // Assuming 1.jpg is in the public folder
        alt="Background"
        fill
        style={{ objectFit: "cover" }}
        priority // Load the background image early
        unoptimized // Useful if the image is served locally
        className="-z-10" // Place the image behind other content
      />
      */}

      {/* Optional: Dark Scrim for Text Readability - REMOVED (Adjust if needed over gradient) */}
      {/* <div className="absolute inset-0 bg-black/40 -z-10"></div> */}

      {/* Status Indicator (Optional but helpful for debugging) */}
      <div className="absolute top-3 right-3 text-xs p-1.5 rounded bg-black/50 backdrop-blur-sm z-20 flex flex-col items-end gap-1">
        <span>WS: {wsStatus}</span>
        {currentSpeaker && <span>Speaker: {currentSpeaker}</span>}
      </div>

      {/* Content Container - Centering the text */}
      <div className="relative z-10 flex flex-col items-center justify-center h-screen p-6 text-center">
        {/* Speaker Name Display (conditional) */}
        {currentSpeaker && (
          <h2
            // Use a key to potentially help with transitions if needed later
            key={`${currentSpeaker}-name`}
            className="text-xl md:text-2xl lg:text-3xl font-medium mb-4 text-gray-100 animate-fade-in" // Adjusted color slightly for contrast
            style={{ textShadow: "1px 1px 4px rgba(0,0,0,0.7)" }} // Increased shadow slightly
          >
            {currentSpeaker} says:
          </h2>
        )}

        {/* Main Text Display (Smaller) */}
        <p
          key={currentText} // Force re-render on text change for animation trigger
          // CHANGED: Reduced font sizes
          className="text-2xl md:text-3xl lg:text-4xl font-semibold animate-fade-in"
          style={{ textShadow: "2px 2px 8px rgba(0,0,0,0.8)" }}
        >
          {currentText}
        </p>
      </div>

      {/* Basic Fade-in Animation Definition (add to your global CSS if preferred) */}
      <style jsx global>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
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
