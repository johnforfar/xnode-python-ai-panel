import { NextRequest, NextResponse } from "next/server";

// Define the base URL of your Python backend where audio is served
const BACKEND_AUDIO_BASE_URL = "http://127.0.0.1:8000/audio";

export async function GET(
  request: NextRequest,
  { params }: { params: { filename: string[] } }
) {
  // params.filename will be an array of path segments
  // e.g., if request is /api/audio/foo/bar.mp3, filename will be ['foo', 'bar.mp3']
  const requestedPath = params.filename.join("/");

  if (!requestedPath) {
    return NextResponse.json(
      { error: "Filename not provided" },
      { status: 400 }
    );
  }

  const backendUrl = `${BACKEND_AUDIO_BASE_URL}/${requestedPath}`;
  console.log(`Proxying audio request for: ${requestedPath} -> ${backendUrl}`);

  try {
    // Fetch the audio file from the backend server-side
    const backendResponse = await fetch(backendUrl, {
      method: "GET",
      headers: {
        // Add any necessary headers if your backend requires them
        // 'Authorization': 'Bearer your_token'
      },
      // Important for fetch caching behavior if needed, default might be okay
      cache: "no-store",
    });

    // Check if the backend responded successfully
    if (!backendResponse.ok) {
      console.error(
        `Backend returned error ${backendResponse.status} for ${backendUrl}`
      );
      // Return the same status code the backend gave (like 404)
      return NextResponse.json(
        { error: `Backend error: ${backendResponse.statusText}` },
        { status: backendResponse.status }
      );
    }

    // Get the audio data as a Blob
    const audioBlob = await backendResponse.blob();

    // Get the content type from the backend response
    const contentType =
      backendResponse.headers.get("Content-Type") || "audio/mpeg"; // Default to audio/mpeg

    // Create a new NextResponse to stream the audio data back to the browser
    const response = new NextResponse(audioBlob, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        // Optional: Add cache control headers if desired
        // 'Cache-Control': 'public, max-age=3600', // Cache for 1 hour example
      },
    });

    return response;
  } catch (error: any) {
    console.error(`Error fetching audio from backend (${backendUrl}):`, error);
    // Add more detail to the error response if possible
    const cause = error.cause
      ? `Cause: ${error.cause.code} ${error.cause.address}:${error.cause.port}`
      : "";
    return NextResponse.json(
      {
        error: "Internal server error proxying audio",
        details: error.message,
        cause,
      },
      { status: 500 }
    );
  }
}

// Optional: Handle other methods if needed, otherwise GET is default
// export async function POST(request: NextRequest) { ... }
