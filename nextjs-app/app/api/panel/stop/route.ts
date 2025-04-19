import { NextResponse } from 'next/server';

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

export async function POST(request: Request) {
  console.log("API Route: POST /api/panel/stop called");
  try {
     const response = await fetch(`${PYTHON_BACKEND_URL}/api/stop`, {
       method: 'POST',
       headers: {
          ...(PYTHON_API_SECRET && { 'Authorization': PYTHON_API_SECRET }),
          // No Content-Type needed if no body
       },
       // No body needed for stop
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("API Route Error (Stop): Backend returned error", response.status, data);
      return NextResponse.json({ error: data.error || 'Failed to stop panel via backend' }, { status: response.status });
    }

    console.log("API Route: POST /api/panel/stop - Success");
    return NextResponse.json(data);

  } catch (error: any) {
    console.error("API Route Error (Stop): Catch block", error);
    return NextResponse.json({ error: 'Internal Server Error stopping panel', details: error.message }, { status: 500 });
  }
}