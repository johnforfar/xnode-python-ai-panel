import { NextRequest, NextResponse } from 'next/server';

// Comment out backend URL/Secret for this test
// const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000';
// const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

const BACKEND_URL = 'http://127.0.0.1:8000'; // Use IPv4 loopback

export async function POST(request: NextRequest) {
  console.log(">>> API Route: POST /api/panel/start called");
  try {
    // No need to parse frontend body anymore, just forward the request

    // --- REMOVE body from fetch ---
    const backendResponse = await fetch(`${BACKEND_URL}/api/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Add any other necessary headers
      },
      // body: JSON.stringify({}), // No body needed
      cache: 'no-store',
    });
    // --- End REMOVE ---

    const responseData = await backendResponse.json().catch(() => ({})); // Attempt to parse JSON, default to {} on failure

    if (!backendResponse.ok) {
      console.error(`<<< API Route: Backend fetch FAILED! Target: ${BACKEND_URL}/api/start, Backend Status: ${backendResponse.status}, Response: ${JSON.stringify(responseData).substring(0, 100)}...`);
      return NextResponse.json(
        { error: responseData.error || 'Failed to start panel on backend', details: responseData.message },
        { status: backendResponse.status }
      );
    }

    console.log(`<<< API Route: Backend start successful. Status: ${backendResponse.status}, Response:`, responseData);
    return NextResponse.json(responseData, { status: backendResponse.status });

  } catch (error: any) {
    console.error(">>> API Route Error in /api/panel/start:", error);
    return NextResponse.json(
      { error: 'Internal Server Error in Next.js proxy', details: error.message },
      { status: 500 }
    );
  }
}