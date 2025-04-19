import { NextResponse } from 'next/server';

// Get Python backend URL and Secret from SERVER-SIDE environment variables
const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET; // Secret is NOT prefixed with NEXT_PUBLIC_

export async function GET(request: Request) {
  console.log("API Route: GET /api/panel/status called");
  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/status`, {
      method: 'GET',
      headers: {
        // Include the secret when calling the Python backend from the server-side
        ...(PYTHON_API_SECRET && { 'Authorization': PYTHON_API_SECRET }),
        'Content-Type': 'application/json',
      },
      // Optional: Increase timeout if backend can be slow
      // next: { revalidate: 0 } // Or cache control if desired
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("API Route Error (Status): Backend returned error", response.status, data);
      return NextResponse.json({ error: data.error || 'Failed to fetch status from backend' }, { status: response.status });
    }

    // console.log("API Route: GET /api/panel/status - Success");
    return NextResponse.json(data);

  } catch (error: any) {
    console.error("API Route Error (Status): Catch block", error);
    return NextResponse.json({ error: 'Internal Server Error fetching status', details: error.message }, { status: 500 });
  }
}