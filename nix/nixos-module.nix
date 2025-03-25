{
  config,
  pkgs,
  lib,
  ...
}:
let
  cfg = config.services.xnode-python-ai-panel;
  xnode-python-xnode-python-ai-panel = pkgs.callPackage ./package.nix { };
in
{
  options = {
    services.xnode-python-xnode-python-ai-panel = {
      enable = lib.mkEnableOption "Enable the python app";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.xnode-python-ai-panel = {
      wantedBy = [ "multi-user.target" ];
      description = "Python App.";
      after = [ "network.target" ];
      serviceConfig = {
        ExecStart = "${lib.getExe xnode-python-ai-panel}";
        DynamicUser = true;
      };
    };

    services.ollama = {
      enable = true;
      loadModels = [
        "deepseek-r1:1.5b"
        "qwen2.5-coder:0.5b"
      ]; # Add any ollama models required
    };
  };
}
