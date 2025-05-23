import { NextResponse } from 'next/server';

// Comment out backend URL/Secret for this test
// const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000';
// const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

export async function POST(request: Request) {
  console.log(`>>> API Route: POST /api/panel/stop called`);
  const targetUrl = `${BACKEND_URL}/api/stop`;

  try {
    console.log(`>>> API Route: Attempting to fetch backend at: ${targetUrl}`);
     const backendResponse = await fetch(targetUrl, {
       method: 'POST',
       // No headers/body needed typically for stop
     });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text().catch(() => 'Could not read backend error response');
      console.error(`<<< API Route: Backend fetch FAILED! Target: ${targetUrl}, Backend Status: ${backendResponse.status}, Response: ${errorText.substring(0, 150)}...`);
      return NextResponse.json(
          { error: `Backend returned status ${backendResponse.status}`, details: errorText.substring(0, 150) },
          { status: 500 }
      );
    }

    const data = await backendResponse.json();
    console.log(`<<< API Route: Backend stop request successful from ${targetUrl}.`);
    return NextResponse.json(data);

  } catch (error: any) {
    const causeCode = error.cause?.code;
    const errorMessage = `<<< API Route: CRITICAL ERROR fetching backend at ${targetUrl}. Cause: ${causeCode || error.message || 'Unknown fetch error'}`;
    console.error(errorMessage);
    return NextResponse.json(
        { error: `Failed to connect to backend service. Is it running?`, cause: causeCode || error.message },
        { status: 500 }
    );
  }
}