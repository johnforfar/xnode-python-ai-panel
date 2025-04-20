import { NextResponse } from 'next/server';

// Comment out backend URL/Secret for this test
// const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
// const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET;

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const PYTHON_API_SECRET = process.env.PYTHON_API_SECRET; // Still read the secret env var

export async function GET(request: Request) {
  console.log("API Route: GET /api/panel/status called (runtime)");
  let responseStatus = 0; // Variable to store backend response status
  try {
    console.log(`API Route: Fetching ${PYTHON_BACKEND_URL}/api/status...`); // Log before fetch
    const response = await fetch(`${PYTHON_BACKEND_URL}/api/status`, {
      method: 'GET',
      headers: {
        // --- Authorization header is STILL commented out ---
        // ...(PYTHON_API_SECRET && { 'Authorization': PYTHON_API_SECRET }),
        'Content-Type': 'application/json',
      },
      cache: 'no-store', // Keep cache disabled for status
    });

    responseStatus = response.status; // Store status code
    console.log(`API Route: Backend fetch completed with status: ${responseStatus}`); // Log after fetch

    const data = await response.json();
    console.log("API Route: Backend response JSON parsed."); // Log after parsing JSON

    // --- Remove the dummy return ---
    // console.log("API Route: Returning dummy status data for build test.");
    // return NextResponse.json({ status: "Build Test OK", ... });

    // --- Return the actual response from the backend ---
    if (!response.ok) {
      console.error("API Route Error (Status): Backend returned non-OK status", responseStatus, data);
      // Forward backend error if possible
      return NextResponse.json({ error: data.error || `Backend error ${responseStatus}` }, { status: responseStatus });
    }
    console.log("API Route: Returning successful response to frontend."); // Log before success return
    return NextResponse.json(data); // Return actual data

  } catch (error: any) {
    console.error(`API Route Error (Status): Catch block (Backend status was ${responseStatus})`, error); // Log error and status code if available
    // Determine appropriate status code for the error response
    const statusCode = (responseStatus >= 400 && responseStatus < 600) ? responseStatus : 500;
    return NextResponse.json({ error: 'Internal Server Error fetching status', details: error.message }, { status: statusCode });
  }
}