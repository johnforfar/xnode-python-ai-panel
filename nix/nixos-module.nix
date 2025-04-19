{
  config,
  pkgs,
  lib,
  ...
}:
let
  cfg = config.services.xnode-python-ai-panel;
  xnode-python-ai-panel-backend = pkgs.callPackage ./python-app.nix { };
  xnode-python-ai-panel-frontend = pkgs.callPackage ./nextjs-app.nix { };
in
{
  options = {
    services.xnode-python-ai-panel = {
      enable = lib.mkEnableOption "Enable the OpenxAI AI Panel services";
      secret = lib.mkOption {
        type = lib.types.str;
        default = "OpenxAI";
        description = ''
          Backend and frontend shared communication secret.
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    # Nextjs frontend service
    users.groups.ai-panel-frontend = { };
    users.users."ai-panel-frontend" = {
      isSystemUser = true;
      group = "ai-panel-frontend";
    };

    systemd.services.xnode-python-ai-panel-frontend = {
      wantedBy = [ "multi-user.target" ];
      description = "OpenxAI Next.js Frontend";
      after = [
        "network.target"
        "xnode-python-ai-panel-backend.service"
      ];
      environment = {
        HOSTNAME = "0.0.0.0";
        PORT = toString 3000;
        SECRET = cfg.secret;
      };
      serviceConfig = {
        ExecStart = "${lib.getExe xnode-python-ai-panel-frontend}";
        User = "ai-panel-frontend";
        Group = "ai-panel-frontend";
        CacheDirectory = "nextjs-app";
      };
    };

    # Python API service
    users.groups.ai-panel-backend = { };
    users.users."ai-panel-backend" = {
      isSystemUser = true;
      group = "ai-panel-backend";
    };

    systemd.tmpfiles.rules = [
      "d /ai-panel-data 0755 ai-panel-backend ai-panel-backend"
    ];

    systemd.services.xnode-python-ai-panel-backend = {
      wantedBy = [ "multi-user.target" ];
      description = "OpenxAI Python API Server";
      after = [
        "network.target"
        "ollama.service"
        #"postgresql.service"
      ];
      environment = {
        PORT = toString 8000;
      #  DB_HOST = "localhost";
      #  DB_PORT = toString config.services.postgresql.settings.port;
      #  DB_NAME = "game";
      #  DB_USER = "game";
      #  DB_PASS = "game";
        SECRET = cfg.secret;
      };
      serviceConfig = {
        ExecStart = "${lib.getExe xnode-python-ai-panel-backend}";
        User = "ai-panel-backend";
        Group = "ai-panel-backend";
        WorkingDirectory = "/ai-panel-data";
        Restart = "on-failure";
      };
    };

    services.ollama = {
      enable = true;
      loadModels = [ "deepseek-r1:1.5b" ];
    };

    #services.postgresql = {
    #  enable = true;
    #  enableTCPIP = true;
    #  initialScript = pkgs.writeText "init-sql-script" ''
    #    CREATE ROLE game WITH LOGIN PASSWORD 'game' CREATEDB;
    #    CREATE DATABASE game;
    #    GRANT ALL PRIVILEGES ON DATABASE game TO game;
    #    \c game postgres
    #    GRANT ALL ON SCHEMA public TO game;
    #  '';
    #};

    # Open firewall ports
    networking.firewall.allowedTCPPorts = [
      3000
    ];
  };
}
