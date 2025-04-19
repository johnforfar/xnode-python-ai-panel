{
  description = "OpenxAI AI Panel";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    {
      self,
      nixpkgs,
      systems,
    }:
    let
      # A helper that helps us define the attributes below for
      # all systems we care about.
      eachSystem =
        f:
        nixpkgs.lib.genAttrs (import systems) (
          system:
          f {
            inherit system;
            pkgs = nixpkgs.legacyPackages.${system};
          }
        );
    in
    {
      packages = eachSystem (
        { pkgs, system, ... }:
        let
          frontend = pkgs.callPackage ./nix/nextjs-app.nix { };
          backend = pkgs.callPackage ./nix/python-app.nix { };

          # Create a more robust script with better error handling
          default = pkgs.writeShellScriptBin "run-openxai" ''
            # Define a function to check if a service is already running on a port
            is_port_in_use() {
              lsof -i :$1 > /dev/null 2>&1
              return $?
            }

            # Start Ollama service if it's not already running
            echo "Checking Ollama service..."
            OLLAMA_RUNNING=false

            if ${pkgs.ollama}/bin/ollama list >/dev/null 2>&1; then
              echo "Ollama is already running"
              OLLAMA_RUNNING=true
            else
              echo "Starting Ollama service..."
              # Try to pull the model first
              ${pkgs.ollama}/bin/ollama pull deepseek-r1:1.5b || echo "Warning: Failed to pull deepseek-r1:1.5b model"
              
              # Start ollama in background
              ${pkgs.ollama}/bin/ollama serve &
              OLLAMA_PID=$!
              
              # Give ollama time to start
              echo "Waiting for Ollama to initialize (10 seconds)..."
              sleep 10
              
              # Check if it's running
              if ${pkgs.ollama}/bin/ollama list >/dev/null 2>&1; then
                echo "Ollama started successfully"
                OLLAMA_RUNNING=true
              else
                echo "Warning: Ollama may not have started correctly, but continuing..."
              fi
            fi

            ## Start PostgreSQL service if it's not already running
            #echo "Checking PostgreSQL service..."
            #if ${pkgs.postgresql}/bin/pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
            #  echo "PostgreSQL is already running"
            #  POSTGRES_RUNNING=true
            #else
            #  echo "Starting PostgreSQL service..."
            #  
            #  # Create data directory if it doesn't exist
            #  PGDATA="/tmp/openxai-pgdata"
            #  mkdir -p $PGDATA
            #  chmod 700 $PGDATA
            #  
            #  # Initialize the database if needed
            #  if [ ! -f "$PGDATA/PG_VERSION" ]; then
            #    echo "Initializing PostgreSQL database..."
            #    ${pkgs.postgresql}/bin/initdb -D $PGDATA
            #    
            #    # Configure PostgreSQL to accept local connections
            #    echo "listen_addresses = 'localhost'" >> $PGDATA/postgresql.conf
            #    echo "port = 5432" >> $PGDATA/postgresql.conf
            #  fi
            #  
            #  # Start PostgreSQL
            #  ${pkgs.postgresql}/bin/pg_ctl -D $PGDATA -o "-k /tmp" start
            #  POSTGRES_PID=$!
            #  
            #  echo "Waiting for PostgreSQL to initialize (5 seconds)..."
            #  sleep 5
            #  
            #  # Check if it's running
            #  if ${pkgs.postgresql}/bin/pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
            #    echo "PostgreSQL started successfully"
            #    POSTGRES_RUNNING=true
            #    
            #    # Create database and user if they don't exist
            #    ${pkgs.postgresql}/bin/createuser -h localhost postgres -s || echo "User postgres may already exist"
            #    ${pkgs.postgresql}/bin/createdb -h localhost -O postgres postgres || echo "Database postgres may already exist"
            #    
            #    # Initialize the database schema
            #    ${pkgs.postgresql}/bin/psql -h localhost -U postgres -d postgres -c "
            #      CREATE TABLE IF NOT EXISTS conversations (
            #          id SERIAL PRIMARY KEY,
            #          user_id TEXT NOT NULL,
            #          message TEXT NOT NULL,
            #          is_user BOOLEAN NOT NULL,
            #          timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            #          conversation_id TEXT NOT NULL,
            #          level INTEGER,
            #          is_winning_move BOOLEAN,
            #          session_id TEXT,
            #          metadata JSONB
            #      );
            #      
            #      CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
            #      CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON conversations(conversation_id);
            #      CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
            #    "
            #  else
            #    echo "Warning: PostgreSQL may not have started correctly, but continuing..."
            #    POSTGRES_RUNNING=false
            #  fi
            #fi

            # Try to start the backend if it's not already running
            echo "Checking backend service..."
            if is_port_in_use 8000; then
              echo "A service is already running on port 8000, assuming it's the backend"
              BACKEND_RUNNING=true
            else
              echo "Starting OpenxAI backend..."
              
              # First try direct path to what we expect to be the executable
              if [ -f "${backend}/bin/openxai-api" ] && [ -x "${backend}/bin/openxai-api" ]; then
                BACKEND_EXE="${backend}/bin/openxai-api"
              else
                # List all files in bin and try to find an executable
                for file in ${backend}/bin/*; do
                  if [ -f "$file" ] && [ -x "$file" ]; then
                    BACKEND_EXE="$file"
                    break
                  fi
                done
              fi
              
              if [ -z "$BACKEND_EXE" ]; then
                echo "Warning: Could not find backend executable in ${backend}/bin"
                echo "Contents of ${backend}/bin:"
                ls -la ${backend}/bin || echo "Failed to list directory contents"
                echo "Continuing without backend..."
                BACKEND_RUNNING=false
              else
                echo "Found backend executable: $BACKEND_EXE"
                
                # ADD ENVIRONMENT VARIABLES HERE
                export DB_HOST=localhost
                export DB_PORT=5432
                export DB_NAME=postgres
                export DB_USER=postgres
                export DB_PASS=postgres
                
                $BACKEND_EXE &
                BACKEND_PID=$!
                echo "Waiting for backend to initialize (5 seconds)..."
                sleep 5
                
                # Check if backend is running
                if is_port_in_use 8000; then
                  echo "Backend started successfully"
                  BACKEND_RUNNING=true
                else
                  echo "Warning: Backend may not have started correctly, but continuing..."
                  BACKEND_RUNNING=false
                fi
              fi
            fi

            # Start the frontend service
            echo "Starting OpenxAI frontend..."

            # First try direct path to what we expect to be the executable
            if [ -f "${frontend}/bin/next-start" ] && [ -x "${frontend}/bin/next-start" ]; then
              FRONTEND_EXE="${frontend}/bin/next-start"
            else
              # List all files in bin and try to find an executable
              for file in ${frontend}/bin/*; do
                if [ -f "$file" ] && [ -x "$file" ]; then
                  FRONTEND_EXE="$file"
                  break
                fi
              done
            fi

            if [ -z "$FRONTEND_EXE" ]; then
              echo "Error: Could not find frontend executable in ${frontend}/bin"
              echo "Contents of ${frontend}/bin:"
              ls -la ${frontend}/bin || echo "Failed to list directory contents"
              echo "Cannot continue without frontend"
              
              # Clean up backend if we started it
              if [ -n "$BACKEND_PID" ]; then
                echo "Cleaning up backend process..."
                kill $BACKEND_PID 2>/dev/null || true
              fi
              
              # Clean up ollama if we started it
              if [ -n "$OLLAMA_PID" ]; then
                echo "Cleaning up ollama process..."
                kill $OLLAMA_PID 2>/dev/null || true
              fi
              
              # Clean up PostgreSQL if we started it
              if [ -n "$POSTGRES_PID" ]; then
                echo "Cleaning up PostgreSQL process..."
                ${pkgs.postgresql}/bin/pg_ctl -D /tmp/openxai-pgdata stop || true
              fi
              
              exit 1
            else
              echo "Found frontend executable: $FRONTEND_EXE"
              $FRONTEND_EXE
              FRONTEND_EXIT=$?
              
              echo "Frontend exited with code $FRONTEND_EXIT"
            fi

            # Clean up processes we started
            if [ -n "$BACKEND_PID" ]; then
              echo "Cleaning up backend process..."
              kill $BACKEND_PID 2>/dev/null || true
            fi

            if [ -n "$OLLAMA_PID" ]; then
              echo "Cleaning up ollama process..."
              kill $OLLAMA_PID 2>/dev/null || true
            fi

            # Clean up PostgreSQL if we started it
            if [ -n "$POSTGRES_PID" ]; then
              echo "Cleaning up PostgreSQL process..."
              ${pkgs.postgresql}/bin/pg_ctl -D /tmp/openxai-pgdata stop || true
            fi

            echo "All services stopped."
          '';
        in
        {
          inherit frontend backend default;
        }
      );

      # Enhanced NixOS module that automates everything
      nixosModules.default = ./nix/nixos-module.nix;
    };
}
