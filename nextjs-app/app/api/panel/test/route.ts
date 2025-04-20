import { NextResponse } from 'next/server';

// Define the backend and Ollama URLs
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://127.0.0.1:11434'; // Default Ollama port

export async function GET(request: Request) {
  console.log(`>>> API Route: GET /api/panel/test called (Combined Backend & Ollama Check)`);
  const backendTestUrl = `${BACKEND_URL}/api/test`;
  const ollamaStatusUrl = `${OLLAMA_URL}/`; // Check the root endpoint

  const backendPromise = fetch(backendTestUrl, { cache: 'no-store', signal: AbortSignal.timeout(3000) });
  const ollamaPromise = fetch(ollamaStatusUrl, { cache: 'no-store', signal: AbortSignal.timeout(2000) }); // Shorter timeout for Ollama

  // Use Promise.allSettled to wait for both checks, regardless of individual success/failure
  const [backendResult, ollamaResult] = await Promise.allSettled([backendPromise, ollamaPromise]);

  let backendStatus: any = { ok: false, error: 'Check not completed' };
  let ollamaStatus: any = { ok: false, message: 'Check not completed', error: 'Unknown' };

  // Process Backend Test Result
  if (backendResult.status === 'fulfilled') {
    const response = backendResult.value;
    if (response.ok) {
      try {
        const data = await response.json();
        backendStatus = { ok: true, data: data };
        console.log(`>>> API Route: Backend test fetched successfully from ${backendTestUrl}.`);
      } catch (parseError: any) {
        backendStatus = { ok: false, error: 'Failed to parse backend JSON response', details: parseError.message };
        console.error(`<<< API Route: Backend test JSON parse FAILED! Target: ${backendTestUrl}`);
      }
    } else {
      const errorText = await response.text().catch(() => 'Could not read backend error response');
      backendStatus = { ok: false, error: `Backend test endpoint returned status ${response.status}`, details: errorText.substring(0, 150) };
      console.error(`<<< API Route: Backend test fetch FAILED! Target: ${backendTestUrl}, Status: ${response.status}, Response: ${errorText.substring(0, 150)}...`);
    }
  } else { // 'rejected'
    const error: any = backendResult.reason;
    const causeCode = error.cause?.code;
    const errorMessage = `CRITICAL ERROR fetching backend test endpoint at ${backendTestUrl}. Cause: ${causeCode || error.name === 'TimeoutError' ? 'Timeout' : error.message || 'Unknown fetch error'}`;
    backendStatus = { ok: false, error: 'Failed to connect to backend test endpoint. Is it running?', cause: causeCode || error.name || error.message };
    console.error(`<<< API Route: ${errorMessage}`);
  }

  // Process Ollama Status Result
  if (ollamaResult.status === 'fulfilled') {
     const response = ollamaResult.value;
     if (response.ok) {
        // Ollama root returns text "Ollama is running" on success
        const text = await response.text();
        if (text.includes("Ollama is running")) {
             ollamaStatus = { ok: true, message: "Ollama is running" };
             console.log(`>>> API Route: Ollama status check successful at ${ollamaStatusUrl}.`);
        } else {
             ollamaStatus = { ok: false, message: "Ollama responded but with unexpected content.", details: text.substring(0, 100)};
             console.warn(`<<< API Route: Ollama check at ${ollamaStatusUrl} returned unexpected text: ${text.substring(0,100)}`);
        }
     } else {
        ollamaStatus = { ok: false, message: `Ollama status endpoint returned status ${response.status}`, error: `HTTP ${response.status}` };
        console.error(`<<< API Route: Ollama status check FAILED! Target: ${ollamaStatusUrl}, Status: ${response.status}`);
     }
  } else { // 'rejected'
     const error: any = ollamaResult.reason;
     const causeCode = error.cause?.code;
     const errorMessage = `CRITICAL ERROR checking Ollama status at ${ollamaStatusUrl}. Cause: ${causeCode || error.name === 'TimeoutError' ? 'Timeout' : error.message || 'Unknown fetch error'}`;
     ollamaStatus = { ok: false, message: 'Failed to connect to Ollama. Is it running?', error: causeCode || error.name || error.message };
     console.error(`<<< API Route: ${errorMessage}`);
  }

  // Return combined results
  // Determine overall status based on whether *both* are ok, or just return individual statuses. Let's return individual.
  console.log(`<<< API Route: Combined test results: Backend OK: ${backendStatus.ok}, Ollama OK: ${ollamaStatus.ok}`);
  return NextResponse.json({
    backend: backendStatus,
    ollama: ollamaStatus,
  });
} 