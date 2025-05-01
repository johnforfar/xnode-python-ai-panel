"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import Image from "next/image";
import { LiveAudioVisualizer } from "react-audio-visualize"; // Re-added import
import { AudioPlayer } from "@/lib/audioplayer";

// Define speaker data mapping - UPDATED based on user request and backend names
const speakerData: { [key: string]: { name: string; image: string } } = {
  "1": { name: "CryptoKitty", image: "/1.jpg" },  // Moderator (previously named Kxi)
  "2": { name: "MrLightning", image: "/2.jpg" },  // Michael Saylor (previously named Liq)
  "3": { name: "PeterGoldBug", image: "/3.jpg" },  // Peter Schiff (previously named Kai)
  "4": { name: "RealSatoshi", image: "/4.jpg" },  // Satoshi Nakamoto (previously named Vivi)
  "5": { name: "TheDon", image: "/5.jpg" },  // Donald Trump (previously named Nn)
};

// --- WebSocket Connection Status Type ---
type WsConnectionStatus =
  | "connecting"
  | "open"
  | "closing"
  | "closed"
  | "error";

export default function SpeakerPage() {
  const params = useParams();
  const speakerId = params?.speakerId as string; // Get speaker ID from route

  const [speakerInfo, setSpeakerInfo] = useState<{
    name: string;
    image: string;
  } | null>(null);
  const [wsStatus, setWsStatus] = useState<WsConnectionStatus>("closed");
  const ws = useRef<WebSocket | null>(null);
  const [audioPlayer, setAudioPlayer] = useState<AudioPlayer | undefined>(
    undefined
  );

  useEffect(() => {
    setAudioPlayer(new AudioPlayer({ muted: true }));
  }, []);

  // Set speaker info based on ID
  useEffect(() => {
    if (speakerId && speakerData[speakerId]) {
      setSpeakerInfo(speakerData[speakerId]);
    } else {
      // Handle invalid/missing speaker ID
      setSpeakerInfo({ name: "Unknown", image: "/default.jpg" }); // Provide a fallback image
    }
  }, [speakerId]);

  // WebSocket Connection Effect
  useEffect(() => {
    if (!speakerId || !audioPlayer) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.hostname}:${
      window.location.protocol === "https:" ? "443" : "8000" // Assuming default ports
    }/ws`;

    console.log(
      `SpeakerPage [${speakerId}]: Attempting WebSocket connection to: ${wsUrl}`
    );
    setWsStatus("connecting");
    const socket = new WebSocket(wsUrl);
    ws.current = socket;

    socket.onopen = () => {
      console.log(
        `SpeakerPage [${speakerId}]: WebSocket connection established`
      );
      setWsStatus("open");
      // Subscribe to my speaker
      [parseInt(speakerId) - 1].forEach((speaker) =>
        socket.send(
          btoa(JSON.stringify({ type: "subscribe", payload: speaker }))
        )
      );
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(atob(event.data));

        // Listen for activity (speaker_activity or audio_update)
        switch (message.type) {
          case "audio":
            audioPlayer.queueFragment(
              message.payload.playAt,
              message.payload.chunk
            );
        }
      } catch (e) {
        console.error(
          `SpeakerPage [${speakerId}]: Failed to parse WebSocket message:`,
          event.data,
          e
        );
      }
    };

    socket.onerror = (event) => {
      console.error(`SpeakerPage [${speakerId}]: WebSocket error:`, event);
      setWsStatus("error");
    };

    socket.onclose = (event) => {
      console.log(
        `SpeakerPage [${speakerId}]: WebSocket connection closed. Code: ${event.code}`
      );
      setWsStatus("closed");
      ws.current = null;
      // Optional: Add reconnect logic here if desired
    };

    // Cleanup function
    return () => {
      console.log(
        `SpeakerPage [${speakerId}]: Cleaning up WebSocket connection.`
      );
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
      setWsStatus("closed");
    };
  }, [speakerId, audioPlayer]);

  const [isPlaying, setIsPlaying] = useState(false);
  useEffect(() => {
    if (!audioPlayer) {
      return;
    }

    setInterval(() => setIsPlaying(audioPlayer.isPlaying()), 100);
  }, [audioPlayer]);

  if (!speakerInfo) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">
        Loading speaker...
      </div>
    );
  }

  return (
    // Use a relative container for absolute positioning of overlays
    <main
      className={`relative min-h-screen w-full font-sans overflow-hidden text-white speaker-background ${
        isPlaying ? "speaking" : ""
      }`}
    >
      {/* Background Image - REMOVED */}
      {/*
      <Image
        src={speakerInfo.image}
        alt="Background"
        fill // Fill the entire main container
        style={{ objectFit: "cover" }} // Cover the area, potentially cropping
        priority
        unoptimized={speakerInfo.image.startsWith("/")}
        sizes="100vw" // Image takes full viewport width
        className="-z-10" // Place the image behind other content
      />
      */}

      {/* Optional: Dark Scrim for Text Readability - REMOVED */}
      {/* <div className="absolute inset-0 bg-black/30 -z-10"></div> */}

      {/* Status indicator & TEMP Toggle Button */}
      <div className="absolute top-3 right-3 text-xs p-1.5 rounded bg-black/50 backdrop-blur-sm z-20 flex flex-col items-end gap-1 status-indicator">
        <span>
          WS: {wsStatus} | ID: {speakerId} | Name: {speakerInfo?.name} |
          Speaking: {isPlaying ? "Yes" : "No"}
        </span>
      </div>

      {/* Content Container (using flex to position items) */}
      <div className="relative z-10 flex flex-col h-screen p-6 md:p-10 speaker-content-container">
        {/* Top Area: Speaker Name */}
        <div className="flex items-start justify-center pt-10 md:pt-16">
          {" "}
          {/* Pushes name down slightly */}
          <h1
            className="text-6xl md:text-7xl font-bold whitespace-normal break-words text-center speaker-name"
            style={{ textShadow: "2px 2px 8px rgba(0,0,0,0.7)" }} // Add shadow for readability
          >
            {speakerInfo.name}
          </h1>
        </div>

        {/* Bottom Area: Visualizer */}
        <div className="grow h-1/4 flex items-center place-items-center justify-center">
          {" "}
          {/* Pushes visualizer up slightly */}
          {/* Conditionally render the LiveAudioVisualizer */}
          {audioPlayer && (
            <LiveAudioVisualizer
              mediaRecorder={audioPlayer.getRecorder()}
              width={500} // Adjust width as needed
              height={75} // Adjust height as needed
              barWidth={3}
              gap={2}
              barColor={"#f7f7f7"} // Example color
            />
          )}
        </div>
      </div>
    </main>
  );
}
