import { NextResponse } from 'next/server';

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

export async function GET(request: Request) {
   console.log("API Route: GET /api/panel/conversation called");
   try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/conversation`, {
       method: 'GET',
       headers: {
          ...(PYTHON_API_SECRET && { 'Authorization': PYTHON_API_SECRET }),
          'Content-Type': 'application/json',
       },
      // next: { revalidate: 0 } // Disable caching if real-time data needed
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("API Route Error (Conversation): Backend returned error", response.status, data);
      return NextResponse.json({ error: data.error || 'Failed to fetch conversation from backend' }, { status: response.status });
    }

    // console.log("API Route: GET /api/panel/conversation - Success");
    return NextResponse.json(data);

  } catch (error: any) {
    console.error("API Route Error (Conversation): Catch block", error);
    return NextResponse.json({ error: 'Internal Server Error fetching conversation', details: error.message }, { status: 500 });
  }
}