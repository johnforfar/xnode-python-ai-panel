{ pkgs, lib }:
pkgs.buildNpmPackage {
  pname = "xnode-python-ai-panel";
  version = "1.0.0";
  src = ../nextjs-app;

  npmDeps = pkgs.importNpmLock {
    npmRoot = ../nextjs-app;
  };
  npmConfigHook = pkgs.importNpmLock.npmConfigHook;

  # Enhanced preBuild that creates an empty globals.css file
  preBuild = ''
    echo "===== Starting build ====="
    echo "Current directory: $(pwd)"
    echo "Files in directory:"
    ls -la
    
    # Create app directory if it doesn't exist
    mkdir -p app
    
    # Create an empty globals.css file (or with minimal content)
    echo "Creating empty globals.css file to satisfy build requirements"
    cat > app/globals.css << 'EOF'
/* Empty globals.css file to satisfy Next.js build */
EOF
    
    echo "Created globals.css:"
    ls -la app/
  '';

  postBuild = ''
    # Add a shebang to the server js file, then patch the shebang to use a
    # nixpkgs nodes binary
    sed -i '1s|^|#!/usr/bin/env node\n|' .next/standalone/server.js
    patchShebangs .next/standalone/server.js
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/{share,bin}

    cp -r .next/standalone $out/share/homepage/
    cp -r public $out/share/homepage/public

    mkdir -p $out/share/homepage/.next
    cp -r .next/static $out/share/homepage/.next/static

    # https://github.com/vercel/next.js/discussions/58864
    ln -s /var/cache/nextjs-app $out/share/homepage/.next/cache

    chmod +x $out/share/homepage/server.js

    # Create a simple wrapper script that sets env vars and calls the server.js
    cat > $out/bin/xnode-python-ai-panel-frontend << EOF
#!/bin/sh
export PORT=3000
export HOST=0.0.0.0
export HOSTNAME=0.0.0.0
export NEXT_PUBLIC_HOST=0.0.0.0
cd $out/share/homepage
exec $out/share/homepage/server.js
EOF

    chmod +x $out/bin/xnode-python-ai-panel-frontend

    runHook postInstall
  '';

  doDist = false;

  meta = {
    mainProgram = "xnode-python-ai-panel-frontend";
  };
}
