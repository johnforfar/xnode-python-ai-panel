{
  # Inputs expected based on the example structure
  python3Packages,
  python3,
  pkgs, # Still needed for lib.getExe
  lib,
}:
python3Packages.buildPythonApplication {
  pname = "xnode-python-ai-panel-backend"; # Keep your project name
  version = "1.0";

  src = ../python-app; # Source directory remains the same

  format = "other"; # We create the executable script manually

  # Keep propagatedBuildInputs minimal, matching the example's approach
  propagatedBuildInputs = with python3.pkgs; [
    setuptools
  ];

  buildInputs = [ pkgs.stdenv.cc.cc.lib ];

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
      pip install -r $out/bin/src/requirements.txt
      export PORT=8000
      python $out/bin/src/app.py
      EOF

      chmod +x $out/bin/xnode-python-ai-panel-backend
    '';

  # No installPhase needed if buildPhase creates everything in $out

  meta = {
    description = "Python backend for AI Panel Discussion";
    mainProgram = "xnode-python-ai-panel-backend";
  };
}
