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

export default function Home() {
  // --- State Declarations ---
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [panelStatus, setPanelStatus] = useState<PanelStatus | null>(null);
  const [numAgents, setNumAgents] = useState<number>(2); // For the input field
  const [isLoading, setIsLoading] = useState(false); // Generic loading state for API calls
  const [error, setError] = useState<string | null>(null); // For displaying errors

  // Autoscroll chat to bottom
  const chatEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => {
    // Only scroll if near the bottom to avoid annoying jumps if user scrolls up
    const scrollContainer = chatEndRef.current?.parentElement?.parentElement; // Adjust selector based on your ScrollArea implementation
     if (scrollContainer) {
       const isNearBottom = scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight < 150;
       if (isNearBottom) {
           chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
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
        const errorData = await response.json().catch(() => ({ error: "Failed to parse error" }));
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.error || 'Unknown status error'}`);
      }
      const data: PanelStatus = await response.json();
      setPanelStatus(data);
      setError(null); // Clear error on success
    } catch (err: any) {
      console.error("Error fetching status:", err);
      setError(`Failed to fetch status: ${err.message}`);
    }
  };

  const fetchConversation = async () => {
     // Fetch conversation from our Next.js API route
    try {
      const response = await fetch(`/api/panel/conversation`); // Call the Next.js API route
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "Failed to parse error" }));
        throw new Error(`HTTP error! status: ${response.status} - ${errorData.error || 'Unknown conversation error'}`);
      }
      const data = await response.json();
      // Ensure data.history is an array before setting
      setConversation(Array.isArray(data.history) ? data.history : []);
      setError(null); // Clear error on success
    } catch (err: any) {
      console.error("Error fetching conversation:", err);
      setError(`Failed to fetch conversation: ${err.message}`);
    }
  };

  const handleStart = async () => {
      if (isLoading || panelStatus?.active) return;
      setIsLoading(true);
      setError(null);
      try {
          // Send start request to our Next.js API route
          const response = await fetch(`/api/panel/start`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ numAgents: numAgents })
          });
          if (!response.ok) {
              const errorData = await response.json().catch(() => ({ error: "Failed to parse error" }));
              throw new Error(`Failed to start: ${response.status} - ${errorData.error || ''}`);
          }
          console.log('Panel start request sent');
          // Fetch status and conversation soon after to reflect changes
          setTimeout(fetchStatus, 300);
          setTimeout(fetchConversation, 500);
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
              method: 'POST',
              // No body needed for stop typically
          });
           if (!response.ok) {
              const errorData = await response.json().catch(() => ({ error: "Failed to parse error" }));
              throw new Error(`Failed to stop: ${response.status} - ${errorData.error || ''}`);
           }
          console.log('Panel stop request sent');
          // Fetch status and conversation soon after
          setTimeout(fetchStatus, 300);
          setTimeout(fetchConversation, 500);
      } catch (err: any) {
          console.error("Error stopping panel:", err);
          setError(`Stop failed: ${err.message}`);
      } finally {
          setIsLoading(false);
      }
  };

  // --- Effects ---

  // Initial fetch and polling
  useEffect(() => {
    fetchStatus();
    fetchConversation(); // Fetch initial data

    const intervalId = setInterval(() => {
        fetchStatus();
        if (panelStatus?.active || conversation.length === 0) {
             fetchConversation();
        }
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(intervalId); // Cleanup interval on unmount
  }, [panelStatus?.active]); // Re-evaluate polling needs if active status changes

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
      <div className={`flex h-8 w-8 items-center justify-center rounded-full ${bgColor} text-white text-sm font-bold shrink-0`}>
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
            <h1 className="text-xl font-bold text-white">AI Panel Discussion</h1>

            {/* Status Display */}
            <div className="text-center">
                <p className={`text-lg font-semibold ${panelStatus?.active ? 'text-green-400' : 'text-yellow-400'}`}>
                    Status: {panelStatus?.status ?? "Loading..."}
                </p>
                {panelStatus && <p className="text-xs text-gray-400">Agents: {panelStatus.num_agents}</p>}
            </div>

            {/* Controls */}
             <div className="flex items-center gap-4">
                 <div className="flex items-center gap-2">
                     <label htmlFor="numAgents" className="text-sm">Agents:</label>
                     <input
                         type="number"
                         id="numAgents"
                         value={numAgents}
                         onChange={(e) => setNumAgents(Math.max(1, Math.min(4, parseInt(e.target.value) || 1)))}
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
                      <div className="flex-1 message-bubble bg-black/20 p-3 rounded-lg min-w-0"> {/* Added min-w-0 for flexbox wrapping */}
                        <div className="text-base text-gray-100 break-words">{msg.text}</div> {/* Allow long words to break */}
                         <div className="text-xs text-gray-500 mt-1 text-right">
                             {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                         </div>
                      </div>
                    </div>
                  </div>
                ))}
                 {isLoading && panelStatus?.active && conversation.length > 0 && ( // Show loading/thinking indicator only when active
                    <div className="mb-4 flex items-start gap-3">
                        <Avatar agent="System" />
                        <div className="flex-1 message-bubble bg-black/20 p-3 rounded-lg italic text-gray-400">
                            Panel is thinking...
                        </div>
                    </div>
                 )}
                 {/* Anchor for scrolling */}
                 <div ref={chatEndRef} style={{ height: "1px" }} />
              </div>
            </ScrollArea>
          </div>
      </div>
    </>
  );
}
