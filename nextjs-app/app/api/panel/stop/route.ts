import { NextResponse } from 'next/server';

// Comment out backend URL/Secret for this test
// const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
// const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

export async function POST(request: Request) {
  console.log("API Route: POST /api/panel/stop called (runtime)");
  try {
     // --- Restore the fetch call ---
     const response = await fetch(`${PYTHON_BACKEND_URL}/api/stop`, {
       method: 'POST',
       headers: {
          // --- Authorization header is STILL commented out ---
          // ...(PYTHON_API_SECRET && { 'Authorization': PYTHON_API_SECRET }),
          // No Content-Type needed if no body
       },
       // No body needed for stop
    });
    const data = await response.json();

    // --- Remove the dummy return ---
    // console.log("API Route: Returning dummy stop success for build test.");
    // return NextResponse.json({ status: "Panel Stop Test OK" });

    // --- Return the actual response from the backend ---
    if (!response.ok) {
      console.error("API Route Error (Stop): Backend returned error", response.status, data);
      return NextResponse.json({ error: data.error || 'Failed to stop panel via backend' }, { status: response.status });
    }
    // Console log kept from before
    console.log("API Route: POST /api/panel/stop - Backend call Success");
    return NextResponse.json(data); // Return actual data

  } catch (error: any) {
    console.error("API Route Error (Stop): Catch block", error);
    return NextResponse.json({ error: 'Internal Server Error stopping panel', details: error.message }, { status: 500 });
  }
}