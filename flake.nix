{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      utils,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        libraryPath = pkgs.lib.makeLibraryPath [
          pkgs.libGL
          pkgs.libglvnd
          pkgs.glib
          pkgs.zlib
          pkgs.libjpeg_turbo
          pkgs.stdenv.cc.cc.lib
        ];
        uvPython = "${pkgs.python311}/bin/python";
        runScript = pkgs.writeShellApplication {
          name = "metrabs-run";
          runtimeInputs = [
            pkgs.uv
            pkgs.python311
          ];
          text = ''
            export LD_LIBRARY_PATH="${libraryPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            export UV_PYTHON="${uvPython}"
            export DATA_ROOT="$PWD/data"
            exec uv run main.py
          '';
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            uv
            python311
            libGL
            zlib
            libjpeg_turbo
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${libraryPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            export UV_PYTHON="${uvPython}"
            export DATA_ROOT="$PWD/data"
          '';
        };

        packages.default = runScript;
        apps.default = {
          type = "app";
          program = "${runScript}/bin/metrabs-run";
        };
      }
    );
}
