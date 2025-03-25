{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
}:
buildPythonPackage rec {
  pname = "uagents";
  version = "0.21.0";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-5rK0vHehqrdQCdOTSAblPOuKgbYhRzORZV8D9cmUugI=";
  };

  # do not run tests
  doCheck = false;

  # specific to buildPythonPackage, see its reference
  pyproject = true;
  build-system = [
    setuptools
  ];
}
