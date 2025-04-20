import { NextResponse } from 'next/server';

// Comment out backend URL/Secret for this test
// const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000';
// const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000';
//const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

// Define the expected backend URL (use environment variable or default)
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

export async function GET(request: Request) {
  console.log(`>>> API Route: GET /api/panel/conversation called`);
  const targetUrl = `${BACKEND_URL}/api/conversation`;

  try {
    console.log(`>>> API Route: Attempting to fetch backend at: ${targetUrl}`);
    // Use no-store cache to ensure fresh data
    const backendResponse = await fetch(targetUrl, { cache: 'no-store' });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text().catch(() => 'Could not read backend error response');
      console.error(`<<< API Route: Backend fetch FAILED! Target: ${targetUrl}, Backend Status: ${backendResponse.status}, Response: ${errorText.substring(0, 150)}...`);
      return NextResponse.json(
          { error: `Backend returned status ${backendResponse.status}`, details: errorText.substring(0, 150) },
          { status: 500 } // Still 500, but with more backend context
      );
    }

    const data = await backendResponse.json();
    console.log(`<<< API Route: Backend conversation fetched successfully from ${targetUrl}.`);
    // Return the success response from the backend
    return NextResponse.json(data); // Assuming backend returns { history: [...] }

  } catch (error: any) {
    const causeCode = error.cause?.code; // Extract the cause code (e.g., ECONNREFUSED)
    const errorMessage = `<<< API Route: CRITICAL ERROR fetching backend at ${targetUrl}. Cause: ${causeCode || error.message || 'Unknown fetch error'}`;
    console.error(errorMessage);

    // Return a clear 500 error to the browser
    return NextResponse.json(
        { error: `Failed to connect to backend service. Is it running?`, cause: causeCode || error.message },
        { status: 500 }
    );
  }
}