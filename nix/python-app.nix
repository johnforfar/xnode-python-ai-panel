{
  python3Packages,
  python3,
  pkgs,
  lib,
}:
python3Packages.buildPythonApplication {
  pname = "xnode-python-ai-panel-backend";
  version = "1.0";

  format = "other";

  propagatedBuildInputs = with python3.pkgs; [
    setuptools
  ];

  buildPhase =
    let
      python = lib.getExe pkgs.python3;
    in
    ''
      # Create an executable script
      mkdir -p $out/bin
      cp -r src $out/bin/src

      cat > $out/bin/xnode-python-ai-panel-backend << EOF
      #!/bin/sh
      ${python} -m venv venv
      source venv/bin/activate
      pip install uagents aiohttp #psycopg2-binary
      export PORT=8000
      python $out/bin/src/app.py
      EOF

      chmod +x $out/bin/xnode-python-ai-panel-backend
    '';

  src = ../python-app;

  meta = {
    mainProgram = "xnode-python-ai-panel-backend";
  };
}
