import { NextResponse } from 'next/server';

// Comment out backend URL/Secret for this test
// const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
// const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

export async function GET(request: Request) {
   console.log("API Route: GET /api/panel/conversation called (runtime)");
   try {
    // --- Restore the fetch call ---
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/conversation`, {
       method: 'GET',
       headers: {
          // --- Authorization header is STILL commented out ---
          // ...(PYTHON_API_SECRET && { 'Authorization': PYTHON_API_SECRET }),
          'Content-Type': 'application/json',
       },
       cache: 'no-store', // Disable caching for conversation for now
    });
    const data = await response.json();

    // --- Remove the dummy return ---
    // console.log("API Route: Returning dummy conversation data for build test.");
    // return NextResponse.json({ history: [] });

    // --- Return the actual response from the backend ---
    if (!response.ok) {
      console.error("API Route Error (Conversation): Backend returned error", response.status, data);
      return NextResponse.json({ error: data.error || 'Failed to fetch conversation from backend' }, { status: response.status });
    }
    return NextResponse.json(data); // Return actual data

  } catch (error: any) {
    console.error("API Route Error (Conversation): Catch block", error);
    return NextResponse.json({ error: 'Internal Server Error fetching conversation', details: error.message }, { status: 500 });
  }
}