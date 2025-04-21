import os
import sys
import asyncio
import logging
from aiohttp import web
import aiohttp_cors
from datetime import datetime
import json
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)
logger.info("--- app.py: Script Start ---")

# --- Import PanelManager INSTANCE, AGENTS list, and websocket_handler from main.py ---
try:
    from main import AGENTS, panel_manager, websocket_handler # Import necessary items
    logger.info("Successfully imported AGENTS, panel_manager instance, and websocket_handler from main.py")
except ImportError as e:
    logger.critical(f"Could not import from main.py. {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"An unexpected error occurred during import from main.py: {e}", exc_info=True)
    sys.exit(1)


# --- API Handlers (Use imported panel_manager instance) ---

async def handle_test(request):
    logger.info("Test endpoint /api/test called successfully.")
    return web.json_response({"status": "ok", "message": "Test successful!"})

async def get_status(request):
    logger.info("GET /api/status")
    try:
        status_data = panel_manager.get_status_data()
        return web.json_response(status_data)
    except Exception as e:
        logger.error(f"Error in get_status handler: {e}", exc_info=True)
        return web.json_response({"error": "Failed to get status"}, status=500)

async def get_conversation(request):
    logger.info("GET /api/conversation")
    try:
        conv_data = panel_manager.get_conversation_data()
        logger.info(f"Returning {len(conv_data['history'])} messages")
        return web.json_response(conv_data)
    except Exception as e:
        logger.error(f"Error in get_conversation handler: {e}", exc_info=True)
        return web.json_response({"error": "Failed to get conversation"}, status=500)


async def handle_start(request):
    logger.info("POST /api/start")
    num_agents = 2 # Default
    try:
        if request.can_read_body and request.content_type == 'application/json':
            data = await request.json()
            num_agents_req = data.get('numAgents', 2)
            # Use len(AGENTS) imported from main
            if not isinstance(num_agents_req, int) or not 1 <= num_agents_req <= len(AGENTS):
                 logger.error(f"Invalid numAgents received: {num_agents_req}")
                 return web.json_response({"error": f"Invalid 'numAgents' (must be 1-{len(AGENTS)})"}, status=400)
            num_agents = num_agents_req
        else:
             logger.info("No valid JSON body for start, using default agents.")

        # Call panel_manager method (which now starts the Bureau)
        success = await panel_manager.start_panel(num_agents)
        if success:
            return web.json_response({"status": "Panel starting (uAgents Bureau)"})
        else:
            # If start_panel returns False, it means it was already running
            return web.json_response({"status": "Already running"}, status=400) # Return 400 Bad Request
    except json.JSONDecodeError:
         logger.error("Failed to parse JSON body for /api/start")
         return web.json_response({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.error(f"Error in handle_start handler: {e}", exc_info=True)
        return web.json_response({"error": "Failed to start panel"}, status=500)


async def handle_stop(request):
    logger.info("POST /api/stop")
    try:
        # Call panel_manager method (which now stops the Bureau)
        success = await panel_manager.stop_panel()
        if success:
            return web.json_response({"status": "Panel stopping"})
        else:
            # If stop_panel returns False, it means it was already stopped
            return web.json_response({"status": "Already stopped"}, status=400) # Return 400 Bad Request
    except Exception as e:
        logger.error(f"Error in handle_stop handler: {e}", exc_info=True)
        return web.json_response({"error": "Failed to stop panel"}, status=500)


# --- Application Setup Function ---
def setup_routes(app):
    logger.info("Setting up routes...")
    cors = aiohttp_cors.setup(app, defaults={
        # Allow the default Next.js port and potentially others
        "*": aiohttp_cors.ResourceOptions( # Changed to wildcard for simplicity, restrict as needed
                allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*",
            ),
    })

    # Helper to add routes
    def add_route(path, handler, method='GET'):
        resource = cors.add(app.router.add_resource(path))
        cors.add(resource.add_route(method, handler))
        logger.info(f"Added route: {method} {path}")

    # Add all API routes
    add_route("/api/test", handle_test)
    add_route("/api/status", get_status)
    add_route("/api/conversation", get_conversation)
    add_route("/api/start", handle_start, method='POST')
    add_route("/api/stop", handle_stop, method='POST')

    # Add WebSocket Route using the handler IMPORTED from main.py
    add_route("/ws", websocket_handler)

    # --- ADD STATIC ROUTE FOR AUDIO FILES ---
    # Determine the path relative to this script's location
    static_audio_path = Path(__file__).parent / 'static' / 'audio'
    # Ensure the directory exists (optional, but good practice)
    static_audio_path.mkdir(parents=True, exist_ok=True)
    app.router.add_static('/audio', path=str(static_audio_path), name='audio', show_index=False) # Set show_index=False
    logger.info(f"Added static route for /audio serving from {static_audio_path}")
    # --- END STATIC ROUTE ---

    logger.info("Routes setup complete.")


# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- app.py: ENTERING __main__ ---")
    logger.info("--- Running app.py in __main__ block ---")
    try:
        app = web.Application()
        # Optional: If handlers needed the app instance later, pass it:
        # app['panel_manager'] = panel_manager # Example if needed

        setup_routes(app) # Setup routes and CORS

        port = int(os.environ.get('PORT', 8000))
        print(f"--- app.py: ABOUT TO CALL web.run_app on port {port} ---")
        logger.info(f"Attempting to run aiohttp app on port {port}")
        web.run_app(app, host='0.0.0.0', port=port) # Use 0.0.0.0
        print("--- app.py: web.run_app FINISHED (should only happen on shutdown) ---")

    except Exception as e:
        print(f"--- app.py: CRITICAL ERROR IN __main__: {e} ---") # Print error directly
        logger.error(f"--- CRITICAL ERROR in app.py __main__: {e} ---", exc_info=True)
        sys.exit(1) # Exit with error code

logger.info("--- app.py: Script End ---") 