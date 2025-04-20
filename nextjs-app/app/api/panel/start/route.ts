import { NextResponse } from 'next/server';

// Comment out backend URL/Secret for this test
// const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
// const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

export async function POST(request: Request) {
  console.log("API Route: POST /api/panel/start called (runtime)");
  try {
     const body = await request.json();
     const numAgents = body.numAgents || 2;

     // --- Restore the fetch call ---
     const response = await fetch(`${PYTHON_BACKEND_URL}/api/start`, {
       method: 'POST',
       headers: {
          // --- Authorization header is STILL commented out ---
          // ...(PYTHON_API_SECRET && { 'Authorization': PYTHON_API_SECRET }),
          'Content-Type': 'application/json',
       },
       body: JSON.stringify({ numAgents: numAgents }),
    });
    const data = await response.json();

    // --- Remove the dummy return ---
    // console.log(`API Route: Returning dummy start success for build test (agents: ${numAgents}).`);
    // return NextResponse.json({ status: "Panel Start Test OK" });

    // --- Return the actual response from the backend ---
    if (!response.ok) {
      console.error("API Route Error (Start): Backend returned error", response.status, data);
      return NextResponse.json({ error: data.error || 'Failed to start panel via backend' }, { status: response.status });
    }
    // Console log kept from before
    console.log("API Route: POST /api/panel/start - Backend call Success");
    return NextResponse.json(data); // Return actual data

  } catch (error: any) {
    console.error("API Route Error (Start): Catch block", error);
     if (error instanceof SyntaxError) {
        return NextResponse.json({ error: 'Invalid JSON in request body' }, { status: 400 });
    }
    return NextResponse.json({ error: 'Internal Server Error starting panel', details: error.message }, { status: 500 });
  }
}