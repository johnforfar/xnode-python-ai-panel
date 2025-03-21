{ python3Packages, python3 }:
python3Packages.buildPythonApplication {
  pname = "xnode-python-ai-panel";
  version = "1.0";

  format = "pyproject";

  propagatedBuildInputs = with python3.pkgs; [ setuptools ];

  src = ../python-app;
}
