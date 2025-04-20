import { NextResponse } from 'next/server';

// Comment out backend URL/Secret for this test
// const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000';
// const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

export async function POST(request: Request) {
  console.log(`>>> API Route: POST /api/panel/start called`);
  const targetUrl = `${BACKEND_URL}/api/start`;
  let requestBody: any;

  try {
     requestBody = await request.json();
     const numAgents = requestBody.numAgents || 2; // Extract numAgents or default
     console.log(`>>> API Route: Attempting to fetch backend at: ${targetUrl} with body:`, requestBody);

     const backendResponse = await fetch(targetUrl, {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ numAgents: numAgents }), // Send necessary data
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
    console.log(`<<< API Route: Backend start request successful from ${targetUrl}.`);
    return NextResponse.json(data);

  } catch (error: any) {
     if (error instanceof SyntaxError) {
        console.error(`<<< API Route: Invalid JSON in request body for ${targetUrl}.`);
        return NextResponse.json({ error: 'Invalid JSON in request body' }, { status: 400 });
     }
    const causeCode = error.cause?.code;
    const errorMessage = `<<< API Route: CRITICAL ERROR fetching backend at ${targetUrl}. Cause: ${causeCode || error.message || 'Unknown fetch error'}`;
    console.error(errorMessage);
    return NextResponse.json(
        { error: `Failed to connect to backend service. Is it running?`, cause: causeCode || error.message },
        { status: 500 }
    );
  }
}