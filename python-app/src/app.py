import os
import sys
import asyncio
import logging
from aiohttp import web
import aiohttp_cors
from datetime import datetime

# --- Logging Setup ---
# Configure logging here or import from a shared logging module if you create one
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)
logger.info("--- app.py: Script Start ---")

# --- Import Core Logic ---
# Import the PanelManager class and create a single instance
try:
    # Assuming PanelManager and AGENTS are defined in main.py
    from main import PanelManager, AGENTS
    logger.info("Successfully imported PanelManager from main.py")
    panel_manager = PanelManager() # Create the single instance
except ImportError as e:
    logger.error(f"Failed to import PanelManager from main.py: {e}. API routes will fail.", exc_info=True)
    # Exit or provide fallback if core logic is essential for server setup
    sys.exit(f"CRITICAL: Could not import PanelManager. Ensure main.py exists and is correct. {e}")

# --- API Handlers (Use the panel_manager instance) ---

async def handle_test(request):
    logger.info("Test endpoint /api/test called successfully.")
    return web.json_response({"status": "ok", "message": "Test successful!"})

async def get_status(request):
    logger.info("GET /api/status")
    try:
        status_data = panel_manager.get_status_data() # Get data from manager instance
        return web.json_response(status_data)
    except Exception as e:
        logger.error(f"Error in get_status handler: {e}", exc_info=True)
        return web.json_response({"error": "Failed to get status"}, status=500)

async def get_conversation(request):
    logger.info("GET /api/conversation")
    try:
        conv_data = panel_manager.get_conversation_data() # Get data from manager instance
        logger.info(f"Returning {len(conv_data['history'])} messages")
        return web.json_response(conv_data)
    except Exception as e:
        logger.error(f"Error in get_conversation handler: {e}", exc_info=True)
        return web.json_response({"error": "Failed to get conversation"}, status=500)


async def handle_start(request):
    logger.info("POST /api/start")
    num_agents = 2 # Default
    try:
        # Check if request has body and content type is json before parsing
        if request.can_read_body and request.content_type == 'application/json':
            data = await request.json()
            num_agents_req = data.get('numAgents', 2)
            # Use the length of AGENTS imported from main
            if not isinstance(num_agents_req, int) or not 1 <= num_agents_req <= len(AGENTS):
                 logger.error(f"Invalid numAgents received: {num_agents_req}")
                 return web.json_response({"error": f"Invalid 'numAgents' (must be 1-{len(AGENTS)})"}, status=400)
            num_agents = num_agents_req
        else:
             logger.info("No valid JSON body for start, using default agents.")

        success = await panel_manager.start_panel(num_agents) # Call manager method
        if success:
            return web.json_response({"status": "Panel starting"})
        else:
            return web.json_response({"status": "Already running"}, status=400)
    except Exception as e:
        logger.error(f"Error in handle_start handler: {e}", exc_info=True)
        return web.json_response({"error": "Failed to start panel"}, status=500)


async def handle_stop(request):
    logger.info("POST /api/stop")
    try:
        success = await panel_manager.stop_panel() # Call manager method
        if success:
            return web.json_response({"status": "Panel stopped"})
        else:
            return web.json_response({"status": "Already stopped"}, status=400)
    except Exception as e:
        logger.error(f"Error in handle_stop handler: {e}", exc_info=True)
        return web.json_response({"error": "Failed to stop panel"}, status=500)


# --- Application Setup Function ---
def setup_routes(app):
    logger.info("Setting up routes...")
    # Use aiohttp_cors for setup
    cors = aiohttp_cors.setup(app, defaults={
        "http://localhost:3000": aiohttp_cors.ResourceOptions(
                allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*",
            ),
        # Add other origins if needed
    })

    # Helper to add routes
    def add_route(path, handler, method='GET'):
        resource = cors.add(app.router.add_resource(path))
        cors.add(resource.add_route(method, handler))
        logger.info(f"Added route: {method} {path}")

    # Add all routes
    add_route("/api/test", handle_test)
    add_route("/api/status", get_status)
    add_route("/api/conversation", get_conversation)
    add_route("/api/start", handle_start, method='POST')
    add_route("/api/stop", handle_stop, method='POST')

    logger.info("Routes setup complete.")


# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- app.py: ENTERING __main__ ---")
    logger.info("--- Running app.py in __main__ block ---")
    try:
        app = web.Application()
        # Optional: Pass panel_manager instance to handlers via app context if preferred
        # app['panel_manager'] = panel_manager

        setup_routes(app) # Setup routes and CORS

        port = int(os.environ.get('PORT', 8000))
        print(f"--- app.py: ABOUT TO CALL web.run_app on port {port} ---")
        logger.info(f"Attempting to run aiohttp app on port {port}")
        web.run_app(app, host='0.0.0.0', port=port) # Use 0.0.0.0
        print("--- app.py: web.run_app FINISHED (should only happen on shutdown) ---")

        logger.info("--- web.run_app finished ---")

    except Exception as e:
        print(f"--- app.py: CRITICAL ERROR IN __main__: {e} ---") # Print error directly
        logger.error(f"--- CRITICAL ERROR in app.py __main__: {e} ---", exc_info=True)
        sys.exit(1) # Exit with error code
else:
    logger.info("--- app.py not running in __main__ block ---")

logger.info("--- app.py: Script End ---") 