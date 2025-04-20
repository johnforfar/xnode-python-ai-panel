import { NextResponse } from 'next/server';

// Define the expected backend URL (use environment variable or default)
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

export async function GET(request: Request) {
  console.log(`>>> API Route: GET /api/panel/test called`);
  const targetUrl = `${BACKEND_URL}/api/test`;

  try {
    console.log(`>>> API Route: Attempting to fetch backend test endpoint at: ${targetUrl}`);
    // Use no-store cache to ensure fresh data
    const backendResponse = await fetch(targetUrl, { cache: 'no-store', signal: AbortSignal.timeout(3000) }); // Add 3s timeout

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text().catch(() => 'Could not read backend error response');
      console.error(`<<< API Route: Backend test fetch FAILED! Target: ${targetUrl}, Backend Status: ${backendResponse.status}, Response: ${errorText.substring(0, 150)}...`);
      return NextResponse.json(
          { error: `Backend test endpoint returned status ${backendResponse.status}`, details: errorText.substring(0, 150) },
          { status: 500 }
      );
    }

    const data = await backendResponse.json();
    console.log(`<<< API Route: Backend test fetched successfully from ${targetUrl}.`);
    // Return the success response from the backend
    return NextResponse.json(data);

  } catch (error: any) {
    const causeCode = error.cause?.code; // Extract the cause code (e.g., ECONNREFUSED)
    const errorMessage = `<<< API Route: CRITICAL ERROR fetching backend test endpoint at ${targetUrl}. Cause: ${causeCode || error.name === 'TimeoutError' ? 'Timeout' : error.message || 'Unknown fetch error'}`;
    console.error(errorMessage);

    // Return a clear 500 error to the browser
    return NextResponse.json(
        { error: `Failed to connect to backend test endpoint. Is it running?`, cause: causeCode || error.name || error.message },
        { status: 500 }
    );
  }
} 