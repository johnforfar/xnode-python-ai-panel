"use client";

import { useState, useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { LiveAudioVisualizer } from "react-audio-visualize";
import { useAudioRecorder } from "react-audio-voice-recorder";
import { Mic, Square } from "lucide-react";

// Define types for conversation messages (Adjust if backend format differs)
interface ConversationMessage {
  timestamp: string;
  agent: string; // e.g., "System", "AI Agent 1", "AI Agent 2"
  address: string; // Identifier ("system", "agent_1", "agent_2")
  text: string;
  audioStatus?: "generating" | "ready" | "failed" | "playing";
  audioUrl?: string | null;
}

// AI Panel Status Type (Adjust based on backend response)
interface PanelStatus {
  status: string;
  active: boolean;
  num_agents: number;
  timestamp: string;
}

// --- Add type for Backend Test result ---
interface BackendTestResult {
  ok: boolean;
  message: string;
  cause?: string;
}

// --- Added: WebSocket Connection Status Type ---
type WsConnectionStatus =
  | "connecting"
  | "open"
  | "closing"
  | "closed"
  | "error";

export default function Home() {
  // --- State Declarations ---
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [panelStatus, setPanelStatus] = useState<PanelStatus | null>(null);
  // --- Add state for backend test ---
  const [backendTest, setBackendTest] = useState<BackendTestResult | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false); // Generic loading state for API calls
  const [error, setError] = useState<string | null>(null); // For displaying errors
  // --- Added: WebSocket State ---
  const [wsStatus, setWsStatus] = useState<WsConnectionStatus>("closed");
  const ws = useRef<WebSocket | null>(null); // Ref to hold the WebSocket instance
  // *** ADD STATE TO GUARD AGAINST STRICT MODE DOUBLE EFFECT RUN ***
  const [wsConnectAttempted, setWsConnectAttempted] = useState(false);
  // --- ADDED TTS State ---
  const [autoPlayAudio, setAutoPlayAudio] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null); // Ref for the single audio player instance
  const [audioPlayer, setAudioPlayer] = useState<MediaRecorder | null>(null);
  const recorder = useAudioRecorder();
  // --- End TTS State ---

  // Autoscroll chat to bottom
  const chatEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    // Only scroll if near the bottom to avoid annoying jumps if user scrolls up
    const scrollContainer = chatEndRef.current?.parentElement?.parentElement; // Adjust selector based on your ScrollArea implementation
    if (scrollContainer) {
      const isNearBottom =
        scrollContainer.scrollHeight -
          scrollContainer.scrollTop -
          scrollContainer.clientHeight <
        150;
      if (isNearBottom) {
        chatEndRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "end",
        });
      }
    } else {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  };

  // --- API Fetching Functions ---
  // These now call the Next.js API routes (the proxy) instead of the Python backend directly

  const fetchStatus = async () => {
    // Fetch status from our Next.js API route
    try {
      const response = await fetch(`/api/panel/status`); // Call the Next.js API route
      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ error: "Failed to parse error" }));
        throw new Error(
          `HTTP error! status: ${response.status} - ${
            errorData.error || "Unknown status error"
          }`
        );
      }
      const data: PanelStatus = await response.json();
      setPanelStatus(data);
      setError(null); // Clear error on success
    } catch (err: any) {
      console.error("Error fetching status:", err);
      setError(`Failed to fetch status: ${err.message}`);
    }
  };

  const fetchInitialConversation = async () => {
    try {
      console.log("Fetching initial conversation history...");
      const response = await fetch(`/api/panel/conversation`);
      if (!response.ok) {
        /* ... error handling ... */ return;
      }
      const data = await response.json();
      setConversation(Array.isArray(data.history) ? data.history : []);
      console.log(
        `Initial history loaded: ${data.history?.length || 0} messages`
      );
      setError(null);
    } catch (err: any) {
      console.error("Error fetching initial conversation:", err);
      setError(`Failed to load history: ${err.message}`);
    }
  };

  const handleStart = async () => {
    if (isLoading || panelStatus?.active) return;
    setIsLoading(true);
    setError(null);
    try {
      // Send start request - no longer need numAgents in body
      const response = await fetch(`/api/panel/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // --- REMOVE body or set to empty object if required by API route ---
        // body: JSON.stringify({}), // Or simply remove the body property
        // --- End REMOVE ---
      });
      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ error: "Failed to parse error" }));
        throw new Error(
          `Failed to start: ${response.status} - ${errorData.error || ""}`
        );
      }
      console.log("Panel start request sent");
      setTimeout(fetchStatus, 300);
      setTimeout(fetchInitialConversation, 500);
    } catch (err: any) {
      console.error("Error starting panel:", err);
      setError(`Start failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    if (isLoading || !panelStatus?.active) return;
    setIsLoading(true);
    setError(null);
    try {
      // Send stop request to our Next.js API route
      const response = await fetch(`/api/panel/stop`, {
        method: "POST",
        // No body needed for stop typically
      });
      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ error: "Failed to parse error" }));
        throw new Error(
          `Failed to stop: ${response.status} - ${errorData.error || ""}`
        );
      }
      console.log("Panel stop request sent");
      // Fetch status and conversation soon after
      setTimeout(fetchStatus, 300);
      setTimeout(fetchInitialConversation, 500);
    } catch (err: any) {
      console.error("Error stopping panel:", err);
      setError(`Stop failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // --- ADDED: Function to fetch backend test status ---
  const fetchBackendTest = async () => {
    console.log("Frontend: Fetching backend test status...");
    setBackendTest(null); // Reset before fetch
    try {
      const response = await fetch(`/api/panel/test`); // Call the new Next.js API route
      const data = await response.json();

      if (!response.ok) {
        // Use error details from the proxied response
        setBackendTest({
          ok: false,
          message: data.error || `HTTP error ${response.status}`,
          cause: data.cause,
        });
        console.error("Frontend: Backend test failed:", data);
      } else {
        // Assuming backend /api/test returns {status: "ok", message: "..."} on success
        setBackendTest({
          ok: data.status === "ok" || response.ok,
          message: data.message || "Backend test responded.",
        });
        console.log("Frontend: Backend test successful:", data);
      }
    } catch (err: any) {
      console.error("Frontend: Error fetching backend test status:", err);
      setBackendTest({
        ok: false,
        message: "Failed to fetch backend test status.",
        cause: err.message,
      });
    }
  };
  // --- End of new function ---

  // --- Add handleClear function ---
  const handleClear = () => {
    console.log("Clearing frontend conversation history.");
    setConversation([]); // Reset the conversation state to an empty array
    // Note: This only clears the frontend view.
    // If you want to clear backend history, you'd need a backend API call here.
  };

  // --- Modified: Play Audio Handler ---
  const handlePlay = (timestamp: string, url: string | null | undefined) => {
    if (!url || !audioRef.current) {
      console.warn("Cannot play audio: No URL or audio element ref.");
      return;
    }

    // --- FIX: Construct the URL pointing to the Next.js proxy API route ---
    // The 'url' from backend is like "/audio/audio_spk0_....mp3"
    // We need to request "/api/audio/audio_spk0_....mp3" from our Next.js app
    // Audio can be fetched directly when running on https (nginx handles the audio path)
    const proxyAudioUrl =
      url.startsWith("/audio/") && window.location.protocol !== "https:"
        ? `/api${url}` // Prepend /api to the existing relative URL
        : url; // If it's already a full URL somehow, use it (fallback)

    // Ensure it starts with a slash if it became relative
    const finalAudioUrl = proxyAudioUrl.startsWith("/")
      ? proxyAudioUrl
      : `/${proxyAudioUrl}`;
    // --- End FIX ---

    console.log(`Playing audio for ${timestamp} via proxy: ${finalAudioUrl}`); // Log the proxy URL

    // Stop any currently playing audio and reset its status
    if (!audioRef.current.paused) {
      audioRef.current.pause();
      setConversation((prev) =>
        prev.map((m) =>
          m.audioStatus === "playing" ? { ...m, audioStatus: "ready" } : m
        )
      );
    }
    audioRef.current.currentTime = 0;

    // Update the status of the message we are about to play
    setConversation((prev) =>
      prev.map((m) =>
        m.timestamp === timestamp ? { ...m, audioStatus: "playing" } : m
      )
    );

    // Set the source to the PROXY URL and play
    audioRef.current.src = finalAudioUrl; // Use the corrected proxy URL
    audioRef.current
      .play()
      .then(async () => {
        let stream: MediaStream | undefined;
        while (stream === undefined || stream.getTracks().length === 0) {
          stream = (
            audioRef.current as any as { captureStream(): MediaStream }
          ).captureStream();
          await new Promise((resolve) => setTimeout(resolve, 100));
        }
        const player = new MediaRecorder(stream);
        player.start();
        setAudioPlayer(player);
      })
      .catch((e) => {
        console.error("Audio play failed:", e);
        setConversation((prev) =>
          prev.map((m) =>
            m.timestamp === timestamp ? { ...m, audioStatus: "failed" } : m
          )
        );
      });

    // --- Add listeners (unchanged logic, but logs will show proxy URL on error) ---
    const currentAudioElement = audioRef.current;

    const onEnded = () => {
      console.log(`Audio ended for ${timestamp}`);
      setConversation((prev) =>
        prev.map((m) =>
          m.timestamp === timestamp ? { ...m, audioStatus: "ready" } : m
        )
      );
      cleanupListeners();
    };
    const onError = () => {
      console.error(`Audio error for ${timestamp}: ${finalAudioUrl}`); // Log proxy URL on error
      setConversation((prev) =>
        prev.map((m) =>
          m.timestamp === timestamp ? { ...m, audioStatus: "failed" } : m
        )
      );
      cleanupListeners();
    };

    const cleanupListeners = () => {
      currentAudioElement?.removeEventListener("ended", onEnded);
      currentAudioElement?.removeEventListener("error", onError);
    };

    // Clear previous listeners before adding new ones
    currentAudioElement?.removeEventListener("ended", onEnded);
    currentAudioElement?.removeEventListener("error", onError);

    currentAudioElement?.addEventListener("ended", onEnded);
    currentAudioElement?.addEventListener("error", onError);
    // --- End Listeners ---
  };
  // --- End Play Audio Handler Modification ---

  // --- Effects ---

  // WebSocket Connection Effect
  useEffect(() => {
    console.log("--- useEffect RUNNING ---");

    // --- Attempt to prevent double-run interference ---
    // If ws.current exists and is not closed, assume connection is active/pending
    // This might happen if the component re-renders but the effect doesn't re-run fully
    if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
      console.log(
        "useEffect: Existing WebSocket found, attaching handlers again if needed or skipping."
      );
      // Potentially re-attach handlers if needed, but often just skipping is fine
      // For now, we'll rely on the initial setup. If issues persist, revisit handler re-attachment.
      return;
    }
    // --- End prevention attempt ---

    // Fetch initial data via HTTP first
    fetchBackendTest();
    fetchStatus();
    fetchInitialConversation();

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.hostname}:${
      window.location.protocol === "https:" ? "443" : "8000"
    }/ws`;
    console.log(`useEffect: Attempting NEW WebSocket connection to: ${wsUrl}`);

    setWsStatus("connecting");
    // Create socket instance local to this effect run
    const socket = new WebSocket(wsUrl);

    // Immediately assign to the ref. The goal is the *second* run in Strict Mode
    // will assign its socket here, making it the "current" one.
    ws.current = socket;
    console.log(`useEffect: Assigned new socket instance to ws.current`);

    socket.onopen = () => {
      console.log(
        `WebSocket onopen event fired (readyState: ${socket.readyState})`
      );
      // Critical check: Only proceed if the ref *still* points to this instance
      if (ws.current === socket) {
        console.log("WebSocket connection established (onopen, ref matches)");
        setWsStatus("open");
        setError(null);
      } else {
        console.warn(
          "WebSocket opened, but ws.current points to a different instance. Closing this orphaned socket."
        );
        socket.close(); // Close the now-orphaned socket
      }
    };

    socket.onmessage = (event) => {
      if (ws.current === socket) {
        try {
          const message = JSON.parse(event.data);
          console.log("WebSocket message received:", message);

          switch (message.type) {
            // --- ADD Case for conversation_history ---
            case "conversation_history":
              if (Array.isArray(message.payload?.history)) {
                console.log(
                  `Received full history: ${message.payload.history.length} messages`
                );
                setConversation(message.payload.history);
              } else {
                console.warn(
                  "Received conversation_history but payload.history was not an array:",
                  message.payload
                );
              }
              break;
            // --- End ADD ---

            // --- ADD Case for moderator_message (handle like agent_message) ---
            case "moderator_message":
              console.log("Handling moderator_message");
              // Treat moderator message similar to agent message for display
              setConversation((prev) => [...prev, message.payload]);
              break;
            // --- End ADD ---

            case "agent_message":
              console.log("Handling agent_message");
              setConversation((prev) => [...prev, message.payload]);
              break;
            case "audio_update":
              setConversation((prev) =>
                prev.map((msg) =>
                  msg.timestamp === message.payload.timestamp
                    ? {
                        ...msg,
                        audioStatus: message.payload.audioStatus,
                        audioUrl: message.payload.audioUrl,
                      }
                    : msg
                )
              );
              break;
            case "status_update":
              setPanelStatus(message.payload);
              break;
            case "system_message":
              setConversation((prev) => [
                ...prev,
                {
                  agent: "System",
                  address: "system",
                  timestamp: new Date().toISOString(),
                  audioStatus: "failed",
                  ...message.payload,
                },
              ]);
              break;
            default:
              // Keep warning for truly unknown types
              console.warn(
                `Received unknown message type: ${message.type}`,
                message
              );
          }
        } catch (e) {
          console.error(
            "Failed to parse WebSocket message or invalid format:",
            event.data,
            e
          );
          setError("Received invalid message from backend.");
        }
      }
    };

    socket.onerror = (event) => {
      console.error(
        `WebSocket onerror event fired (readyState: ${socket.readyState})`
      );
      // Check if this socket is still the one in the ref OR if the ref is null (meaning the intended one failed)
      if (ws.current === socket || ws.current === null) {
        console.error("WebSocket connection error occurred.");
        setWsStatus("error");
        setError("WebSocket connection error.");
        if (ws.current === socket) ws.current = null; // Clear ref if it was this one
      } else {
        console.warn("WebSocket error on non-current socket instance.");
      }
    };

    socket.onclose = (event) => {
      console.log(
        `WebSocket onclose event fired (readyState: ${socket.readyState}). Code: ${event.code}`
      );
      // Check if the ref points to the socket that just closed
      if (ws.current === socket) {
        console.log("Active WebSocket connection closed.");
        setWsStatus("closed");
        if (panelStatus?.active) {
          setPanelStatus((prev) =>
            prev ? { ...prev, status: "Disconnected", active: false } : null
          );
        }
        ws.current = null; // Clear the ref
      } else {
        console.log("Non-active WebSocket instance closed.");
      }
    };

    // Cleanup function
    return () => {
      console.log("--- useEffect CLEANUP running ---");
      // This cleanup runs for *every* execution of the effect function.
      // In Strict Mode, it runs after the first mount, then the second mount runs,
      // then it runs again when the component *actually* unmounts.

      // Check if the ref currently points to the socket created in *this* effect run.
      if (ws.current === socket) {
        console.log(
          "Cleanup: ws.current matches this effect's socket. Preparing to potentially close for UNMOUNT."
        );
        // If the ref points here, it might be the actual unmount cleanup.
        // We'll close it and clear the ref.
        if (
          socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING
        ) {
          console.log(
            `Cleanup: Closing socket instance (readyState: ${socket.readyState}) for unmount.`
          );
          socket.close();
        }
        ws.current = null;
        setWsStatus("closed"); // Ensure status is reset on unmount
      } else {
        // If the ref points elsewhere (likely to the socket from the *second* strict mode run),
        // then this cleanup is for the *first* strict mode run. We should close the socket
        // local to *this* scope, but NOT clear the main ws.current ref.
        console.log(
          "Cleanup: ws.current does NOT match this effect's socket (Likely strict mode cleanup). Closing local socket."
        );
        if (
          socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING
        ) {
          console.log(
            `Cleanup: Closing STRICT MODE socket instance (readyState: ${socket.readyState})`
          );
          socket.close();
        }
      }
    };
  }, []); // Keep empty dependency array

  // Scroll to bottom when conversation updates
  useEffect(() => {
    // Give time for new message to render before scrolling
    const timer = setTimeout(scrollToBottom, 100);
    return () => clearTimeout(timer);
  }, [conversation]); // Trigger on conversation change

  // --- Auto-Play Effect ---
  useEffect(() => {
    if (!autoPlayAudio) return;

    // Find the *last* message in the array that just became 'ready'
    // This avoids playing older messages if multiple become ready at once
    const lastMessage = conversation[conversation.length - 1];

    if (
      lastMessage &&
      lastMessage.audioStatus === "ready" &&
      lastMessage.audioUrl &&
      lastMessage.agent !== "System"
    ) {
      // Check if it was *previously* not ready (to avoid replaying on unrelated updates)
      // This requires comparing with previous state, which is complex here.
      // Simpler approach: Play if it's ready and hasn't been played automatically yet.
      // We need a way to mark it as 'auto-played' or rely on the 'playing' state transition.
      // Let's try playing it directly if it's the last message and is 'ready'.
      console.log(
        `Auto-play triggered for last message [${lastMessage.timestamp}]`
      );
      handlePlay(lastMessage.timestamp, lastMessage.audioUrl);
    }
  }, [conversation, autoPlayAudio]); // Re-run when conversation or toggle changes
  // --- End Auto-Play Effect ---

  // --- UI Components --- (Simplified Avatar)
  const Avatar = ({ agent }: { agent: string }) => {
    // Basic avatar based on agent type
    let bgColor = "bg-gray-500"; // Default
    let initials = agent.substring(0, 1).toUpperCase();
    if (agent === "System") {
      bgColor = "bg-purple-600";
      initials = "S";
    } else if (agent.startsWith("AI")) {
      bgColor = "bg-green-600";
      initials = agent.replace("AI Panel", "P").replace("AI Agent", "A"); // Example initials
    }
    return (
      <div
        className={`flex h-8 w-8 items-center justify-center rounded-full ${bgColor} text-white text-sm font-bold shrink-0`}
      >
        {initials}
      </div>
    );
  };

  // --- NEW: Audio Status Icon Component ---
  const AudioStatusIcon = ({
    status,
    url,
    timestamp,
  }: {
    status?: string;
    url?: string | null;
    timestamp: string;
  }) => {
    if (!status || status === "failed") {
      return (
        <span title="Audio generation failed" className="text-red-500 ml-2">
          ❌
        </span>
      );
    }
    if (status === "generating") {
      return (
        <span
          title="Generating audio..."
          className="text-yellow-500 ml-2 animate-pulse"
        >
          ⏳
        </span>
      );
    }
    if (status === "playing") {
      return (
        <span title="Playing audio..." className="text-blue-400 ml-2">
          🔊
        </span>
      );
    }
    if (status === "ready" && url) {
      return (
        <button
          title="Play audio"
          onClick={() => handlePlay(timestamp, url)}
          className="ml-2 text-green-400 hover:text-green-300 transition-colors"
        >
          ▶️
        </button>
      );
    }
    return null; // No icon if status is unknown or not applicable
  };
  // --- End Audio Status Icon ---

  // --- Render ---
  return (
    <>
      {/* --- ADD Hidden Audio Player --- */}
      <audio ref={audioRef} style={{ display: "none" }} />
      {/* --- End Hidden Audio Player --- */}

      {/* Main container */}
      <div className="flex flex-col min-h-screen bg-gradient-to-b from-[#1B2538] to-[#0F172A] text-white">
        {/* Header */}
        <header className="w-full border-b border-[#454545] py-4 bg-[#0F172A] sticky top-0 z-10 px-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            {/* Title */}
            <h1 className="text-xl font-bold text-white">
              AI Panel Discussion
            </h1>

            {/* Status Display */}
            <div className="text-center">
              {/* --- Display Backend Test Status --- */}
              <div className="text-xs mb-1">
                {backendTest === null && (
                  <span className="text-gray-400">
                    Backend Test: Checking...
                  </span>
                )}
                {backendTest?.ok && (
                  <span className="text-green-400">Backend Test: OK</span>
                )}
                {backendTest && !backendTest.ok && (
                  <span
                    className="text-red-400"
                    title={`Cause: ${backendTest.cause}`}
                  >
                    Backend Test: FAIL
                  </span>
                )}
              </div>
              {/* --- Added: Display WebSocket Status --- */}
              <div className="text-xs mb-1">
                <span
                  className={`
                         ${wsStatus === "open" ? "text-green-400" : ""}
                         ${wsStatus === "connecting" ? "text-yellow-400" : ""}
                         ${
                           wsStatus === "closed" || wsStatus === "error"
                             ? "text-red-400"
                             : ""
                         }
                     `}
                >
                  WS: {wsStatus}
                </span>
              </div>
              <p
                className={`text-lg font-semibold ${
                  panelStatus?.active ? "text-green-400" : "text-yellow-400"
                }`}
              >
                Status: {panelStatus?.status ?? "Loading..."}
              </p>
              {panelStatus && (
                <p className="text-xs text-gray-400">
                  Debaters: {panelStatus.num_agents}
                </p>
              )}
            </div>

            {/* Controls */}
            <div className="flex items-center gap-4">
              <button
                onClick={handleStart}
                disabled={isLoading || panelStatus?.active}
                className="px-4 py-2 bg-green-600 rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Start
              </button>
              <button
                onClick={handleStop}
                disabled={isLoading || !panelStatus?.active}
                className="px-4 py-2 bg-red-600 rounded hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Stop
              </button>
              <button
                onClick={handleClear}
                disabled={conversation.length === 0} // Disable if already empty
                className="px-4 py-2 bg-gray-600 rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Clear
              </button>
              {/* --- ADD Auto-Play Toggle --- */}
              <div className="flex items-center space-x-2">
                <Switch
                  id="autoplay-audio"
                  checked={autoPlayAudio}
                  onCheckedChange={setAutoPlayAudio}
                  aria-label="Auto-play audio"
                />
                <Label
                  htmlFor="autoplay-audio"
                  className="text-sm cursor-pointer"
                >
                  Auto-Play Audio
                </Label>
              </div>
              {/* --- End Auto-Play Toggle --- */}
            </div>
          </div>
          {/* Error Display */}
          {error && (
            <div className="text-center text-red-400 text-sm mt-2 bg-red-900/30 border border-red-500/50 p-2 rounded">
              Error: {error}
            </div>
          )}
        </header>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex flex-col w-full place-items-center m-2">
            <button
              onClick={() =>
                recorder.isRecording
                  ? recorder.stopRecording()
                  : recorder.startRecording()
              }
            >
              {recorder.isRecording ? <Square /> : <Mic />}
            </button>
            {recorder.mediaRecorder && (
              <LiveAudioVisualizer
                mediaRecorder={recorder.mediaRecorder}
                width={500}
                height={50}
              />
            )}
            {audioPlayer && (
              <LiveAudioVisualizer
                mediaRecorder={audioPlayer}
                width={500}
                height={50}
              />
            )}
          </div>
          {/* Messages container */}
          <ScrollArea className="flex-1">
            <div className="px-6 py-4">
              {conversation.map((msg, index) => (
                <div
                  key={`${msg.timestamp}-${index}-${msg.agent}`} // Use a unique key
                  className="mb-4"
                >
                  <div className="flex items-start gap-3">
                    <Avatar agent={msg.agent} />
                    <div className="flex-1 message-bubble bg-black/20 p-3 rounded-lg min-w-0">
                      {" "}
                      {/* Added min-w-0 for flexbox wrapping */}
                      <div className="text-base text-gray-100 break-words">
                        {msg.text}
                      </div>{" "}
                      {/* Allow long words to break */}
                      <div className="flex justify-end items-center text-xs text-gray-500 mt-1">
                        <span>
                          {new Date(msg.timestamp).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                            second: "2-digit",
                          })}
                        </span>
                        {/* Conditionally render icon only for non-system messages */}
                        {msg.agent !== "System" && (
                          <AudioStatusIcon
                            status={msg.audioStatus}
                            url={msg.audioUrl}
                            timestamp={msg.timestamp}
                          />
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {/* Anchor for scrolling */}
              <div ref={chatEndRef} style={{ height: "1px" }} />
            </div>
          </ScrollArea>
        </div>
      </div>
    </>
  );
}
