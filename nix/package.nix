{
  python3Packages,
  python3,
  pkgs,
}:
python3Packages.buildPythonApplication {
  pname = "xnode-python-ai-panel";
  version = "1.0";

  format = "pyproject";

  dependencies = [
    (pkgs.callPackage ./uagents.nix {
      buildPythonPackage = python3Packages.buildPythonPackage;
      setuptools = python3.pkgs.setuptools;
    })
  ];

  propagatedBuildInputs = with python3.pkgs; [
    setuptools
    aiohttp
  ];

  src = ../python-app;
}
