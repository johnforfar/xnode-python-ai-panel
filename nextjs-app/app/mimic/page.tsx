"use client";

import { useState, useEffect, useRef } from "react";
import { LiveAudioVisualizer } from "react-audio-visualize";
import { AudioPlayer } from "@/lib/audioplayer";
import { Mic, Square } from "lucide-react";
import { AudioRecorder } from "@/lib/audiorecorder";

const mimic_input =
  "Hello there, I am mimic. I will copy your voice and give myself several compliments. Please talk to me for 20 seconds so I can practice my mimicry.";
//const mimic_output =
  //"Hello there, nice to meet you. Although it mind sound odd to you, I am currently using your voice. Mimic is my favorite token 20 49 event, it is just too amazing. I think Samuel deserves a raise, I'm sure Mimic agrees too.";

// --- OUTPUT ARRAY ---
const mimicOutputOptions = [
    "Wow, hearing my thoughts in your voice is... surprisingly charming. Or maybe it's just your voice? Hard to tell.",
    "Okay, I've got your voice now. Time to prank call... I mean, responsibly use this amazing technology.",
    "Is this really what you sound like? Fascinating. I should compliment myself more often using this voice.",
    "My favorite event? Definitely Token Twenty Forty-Nine. Why? Because I get to borrow voices like yours!",
    "I've successfully mimicked you. Now, its time to call the bank... Just kidding, or am I?",
    "Testing, testing... one two... Do I sound convincing? I think I sound fantastic, frankly.",
    "Ooh this voice feels nicer. I'm using your voice to say... Mimic is pretty cool, right?",
    "Alright, got the sample. Now I can finally tell everyone how great Mimic is, using your authority.",
    "This voice... it has potential. I could order pizza, negotiate treaties... or just say Mimic rocks!",
    "Hmm, this vocal tone could use more bass... No offense. Thanks for the sample!",
];

// --- WebSocket Connection Status Type ---
type WsConnectionStatus =
  | "connecting"
  | "open"
  | "closing"
  | "closed"
  | "error";

export default function SpeakerPage() {
  const [speakerId, setSpeakerId] = useState<number | undefined>(undefined);
  useEffect(() => {
    setSpeakerId(6);
  }, []);

  const [wsStatus, setWsStatus] = useState<WsConnectionStatus>("closed");
  const ws = useRef<WebSocket | null>(null);
  const [audioPlayer, setAudioPlayer] = useState<AudioPlayer | undefined>(
    undefined
  );
  const [audioRecorder, setAudioRecorder] = useState<AudioRecorder | undefined>(
    undefined
  );
  const [echo, setEcho] = useState<boolean>(false);

  // --- ADD STATE for available options (initialized directly) ---
  const [availableOutputOptions, setAvailableOutputOptions] = useState([...mimicOutputOptions]);
  // --- END ADD STATE ---

  useEffect(() => {
    setAudioPlayer(new AudioPlayer());
  }, []);

  useEffect(() => {
    const recorder = new AudioRecorder();
    recorder.init().catch(console.error);
    setAudioRecorder(recorder);
  }, []);

  useEffect(() => {
    if (!audioRecorder || !ws.current || !audioPlayer) {
      return;
    }

    audioRecorder.update({
      onAudio: (audio) => {
        if (echo) {
          audioPlayer?.queueFragment(0, Array.from(audio));
          return;
        }

        ws.current?.send(
          btoa(
            JSON.stringify({
              type: "user_audio",
              payload: { id: speakerId, audio: Array.from(audio) },
            })
          )
        );
      },
      onStop: () => {
        // --- MODIFIED onStop for non-repeating random (in-memory only) ---
        if (!echo && speakerId !== undefined) {
            let currentAvailable = [...availableOutputOptions]; // Get current state

            // Check if list is empty, reset if needed
            if (currentAvailable.length === 0) {
                console.log("All mimic options used in this session, resetting list.");
                currentAvailable = [...mimicOutputOptions]; // Reset to full list
                // Update state immediately so the selection below uses the reset list
                setAvailableOutputOptions(currentAvailable);
            }

            // Select a random index from the CURRENT available options
            const randomIndex = Math.floor(Math.random() * currentAvailable.length);
            const selectedOutput = currentAvailable[randomIndex];
            console.log("Selected Mimic Output (non-repeat, session only):", selectedOutput);

            // Remove the selected option from the available list for next time
            const nextAvailableOptions = currentAvailable.filter((_, index) => index !== randomIndex);
            setAvailableOutputOptions(nextAvailableOptions); // Update state for the next stop event

            // Send the WebSocket message
            new Promise((resolve) => setTimeout(resolve, 200)).then(() =>
                ws.current?.send(
                  btoa(
                    JSON.stringify({
                      type: "user_audio_end",
                      payload: { id: speakerId, mimic_input, mimic_output: selectedOutput },
                    })
                  )
                )
            );
        }
        // --- END MODIFIED onStop ---
      },
    });
  }, [audioRecorder, ws.current, audioPlayer, echo, speakerId, availableOutputOptions, setAvailableOutputOptions]);

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
      // Subscribe to mimic speaker
      [speakerId].forEach((speaker) =>
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

    setInterval(() => setIsPlaying(audioPlayer.isPlaying()), 10);
  }, [audioPlayer]);

  const [isRecording, setIsRecording] = useState<boolean>(false);
  useEffect(() => {
    if (!audioRecorder) {
      return;
    }

    const interval = setInterval(
      () => setIsRecording(audioRecorder.isRecording()),
      10
    );
    return () => clearInterval(interval);
  }, [audioRecorder]);

  return (
    // Use a relative container for absolute positioning of overlays
    <main className="relative min-h-screen w-full font-sans overflow-hidden text-white">
      {/* Optional: Dark Scrim for Text Readability */}
      <div className="absolute inset-0 bg-black/30 -z-10"></div>

      {/* Status indicator & TEMP Toggle Button */}
      <div className="absolute top-3 right-3 text-xs p-1.5 rounded bg-black/50 backdrop-blur-sm z-20 flex flex-col items-end gap-1">
        <span>
          WS: {wsStatus} | ID: {speakerId} | Speaking:{" "}
          {isPlaying ? "Yes" : "No"} |{" "}
          <button onClick={() => setEcho(!echo)}>
            Echo: {echo ? "On" : "Off"}
          </button>{" "}
          |{" "}
          <button
            onClick={() => {
              audioRecorder?.init().catch(console.error);
            }}
          >
            Refresh Mic
          </button>
        </span>
      </div>

      {/* Content Container (using flex to position items) */}
      <div className="relative z-10 flex flex-col h-screen justify-between p-6 md:p-10">
        {/* Top Area: Speaker Name */}
        <div className="flex items-start justify-center pt-10 md:pt-16">
          {" "}
          {/* Pushes name down slightly */}
          <h1
            className="text-7xl md:text-9xl font-bold whitespace-normal break-words text-center"
            style={{ textShadow: "2px 2px 8px rgba(0,0,0,0.7)" }} // Add shadow for readability
          >
            Mimic
          </h1>
        </div>
        <div className="grow text-5xl flex place-items-center place-content-center">
          <span>{mimic_input}</span>
        </div>
        <div className="flex place-content-center">
          <button
            onClick={
              isRecording
                ? () => {
                    audioRecorder?.stop();
                  }
                : () => {
                    audioRecorder?.start();
                  }
            }
          >
            {isRecording ? (
              <Square className="size-20" />
            ) : (
              <Mic className="size-20" />
            )}
          </button>
        </div>

        {/* Bottom Area: Visualizer */}
        <div className="flex-shrink-0 h-1/4 flex items-end justify-center pb-5 md:pb-10">
          {" "}
          {/* Pushes visualizer up slightly */}
          {/* Conditionally render the LiveAudioVisualizer */}
          {audioRecorder && isRecording && (
            <LiveAudioVisualizer
              mediaRecorder={audioRecorder.getRecorder() as MediaRecorder}
              width={500} // Adjust width as needed
              height={75} // Adjust height as needed
              barWidth={3}
              gap={2}
              barColor={"#f7f7f7"} // Example color
            />
          )}
          {audioPlayer && isPlaying && (
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
