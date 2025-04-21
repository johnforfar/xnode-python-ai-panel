"use client";

import { useState, useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";

// Define types for conversation messages (Adjust if backend format differs)
interface ConversationMessage {
  timestamp: string;
  agent: string; // e.g., "System", "AI Agent 1", "AI Agent 2"
  address: string; // Identifier ("system", "agent_1", "agent_2")
  text: string;
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
  const [numAgents, setNumAgents] = useState<number>(2); // For the input field
  const [isLoading, setIsLoading] = useState(false); // Generic loading state for API calls
  const [error, setError] = useState<string | null>(null); // For displaying errors
  // --- Added: WebSocket State ---
  const [wsStatus, setWsStatus] = useState<WsConnectionStatus>("closed");
  const ws = useRef<WebSocket | null>(null); // Ref to hold the WebSocket instance
  // *** ADD STATE TO GUARD AGAINST STRICT MODE DOUBLE EFFECT RUN ***
  const [wsConnectAttempted, setWsConnectAttempted] = useState(false);

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
      // Send start request to our Next.js API route
      const response = await fetch(`/api/panel/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ numAgents: numAgents }),
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
      // Fetch status and conversation soon after to reflect changes
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
      // Check if this socket is still the one in the ref
      if (ws.current === socket) {
        try {
          const message = JSON.parse(event.data);
          console.log("WebSocket message received:", message);
          // Handle different message types
          switch (message.type) {
            case "agent_message":
              setConversation((prev) => [...prev, message.payload]);
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
                  ...message.payload,
                },
              ]);
              break;
            default:
              console.warn("Received unknown message type:", message.type);
          }
        } catch (e) {
          console.error(
            "Failed to parse WebSocket message or invalid format:",
            event.data,
            e
          );
          setError("Received invalid message from backend.");
        }
      } else {
        // console.log("WebSocket message received on non-current socket instance.");
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

  // --- Render ---
  return (
    <>
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
                  Agents: {panelStatus.num_agents}
                </p>
              )}
            </div>

            {/* Controls */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label htmlFor="numAgents" className="text-sm">
                  Agents:
                </label>
                <input
                  type="number"
                  id="numAgents"
                  value={numAgents}
                  onChange={(e) =>
                    setNumAgents(
                      Math.max(1, Math.min(4, parseInt(e.target.value) || 1))
                    )
                  }
                  min="1"
                  max="4"
                  className="w-16 bg-[#1A1A1A] border border-[#454545] rounded px-2 py-1 text-white"
                  disabled={panelStatus?.active || isLoading}
                />
              </div>
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
                      <div className="text-xs text-gray-500 mt-1 text-right">
                        {new Date(msg.timestamp).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })}
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
