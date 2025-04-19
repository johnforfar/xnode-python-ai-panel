import { NextResponse } from 'next/server';

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

export async function POST(request: Request) {
  console.log("API Route: POST /api/panel/start called");
  try {
     const body = await request.json(); // Get numAgents from frontend request
     const numAgents = body.numAgents || 2; // Default if not provided

     const response = await fetch(`${PYTHON_BACKEND_URL}/api/start`, {
       method: 'POST',
       headers: {
          ...(PYTHON_API_SECRET && { 'Authorization': PYTHON_API_SECRET }),
          'Content-Type': 'application/json',
       },
       body: JSON.stringify({ numAgents: numAgents }), // Forward payload
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("API Route Error (Start): Backend returned error", response.status, data);
      return NextResponse.json({ error: data.error || 'Failed to start panel via backend' }, { status: response.status });
    }

    console.log("API Route: POST /api/panel/start - Success");
    return NextResponse.json(data);

  } catch (error: any) {
    console.error("API Route Error (Start): Catch block", error);
    // Check if error is due to invalid JSON in request body
     if (error instanceof SyntaxError) {
        return NextResponse.json({ error: 'Invalid JSON in request body' }, { status: 400 });
    }
    return NextResponse.json({ error: 'Internal Server Error starting panel', details: error.message }, { status: 500 });
  }
}