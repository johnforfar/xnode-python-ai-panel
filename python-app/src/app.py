import os
import sys
import logging
from aiohttp import web
import aiohttp_cors
import json
from pathlib import Path
from env import data_dir
from main import panel_manager, websocket_handler

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

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
    try:
        if panel_manager.active:
            logger.warning("Start requested but panel already active.")
            return web.json_response({"status": "already_active", "message": "Panel is already running."}, status=400)

        success = await panel_manager.start_panel()
        if success:
            logger.info("Panel started successfully via API request.")
            return web.json_response({"status": "started", "message": "Panel started successfully."})
        else:
            logger.error("PanelManager failed to start.")
            return web.json_response({"error": "Failed to start panel"}, status=500)

    except json.JSONDecodeError:
        logger.error("Bad request: Invalid JSON in /api/start")
        return web.json_response({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        logger.error(f"Error in handle_start handler: {e}", exc_info=True)
        return web.json_response({"error": f"Internal server error: {e}"}, status=500)


async def handle_stop(request):
    logger.info("POST /api/stop")
    try:
        if not panel_manager.active:
            logger.warning("Stop requested but panel not active.")
            return web.json_response({"status": "already_stopped", "message": "Panel is not running."}, status=400)

        success = await panel_manager.stop_panel()
        if success:
            logger.info("Panel stopped successfully via API request.")
            return web.json_response({"status": "stopped", "message": "Panel stopped successfully."})
        else:
            logger.error("PanelManager failed to stop.")
            return web.json_response({"error": "Failed to stop panel"}, status=500)
    except Exception as e:
        logger.error(f"Error in handle_stop handler: {e}", exc_info=True)
        return web.json_response({"error": f"Internal server error: {e}"}, status=500)


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
    static_audio_path = Path(data_dir()) / 'static' / 'audio'
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