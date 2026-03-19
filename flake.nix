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
        python = pkgs.python311;
        nativeRuntimeDeps = with pkgs; [
          dbus
          fontconfig
          freetype
          glib
          libGL
          libglvnd
          libjpeg_turbo
          libxkbcommon
          qt5.qtbase
          qt5.qtwayland
          stdenv.cc.cc.lib
          wayland
          xorg.libICE
          xorg.libSM
          xorg.libX11
          xorg.libXcursor
          xorg.libXdamage
          xorg.libXext
          xorg.libXfixes
          xorg.libXi
          xorg.libXinerama
          xorg.libXrandr
          xorg.libXrender
          xorg.libXtst
          xorg.libxcb
          xorg.xcbutil
          xorg.xcbutilimage
          xorg.xcbutilkeysyms
          xorg.xcbutilrenderutil
          xorg.xcbutilwm
          zlib
        ];
        libraryPath = pkgs.lib.makeLibraryPath nativeRuntimeDeps;
        uvPython = "${python}/bin/python";
        envScript = ''
          export LD_LIBRARY_PATH="${libraryPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          export QT_QPA_PLATFORM="xcb"
          export UV_PYTHON="${uvPython}"
        '';
        runScript = pkgs.writeShellApplication {
          name = "metrabs-run";
          runtimeInputs = [
            pkgs.uv
            python
          ];
          text = ''
            ${envScript}
            exec uv run main.py "$@"
          '';
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            python
          ] ++ nativeRuntimeDeps;

          shellHook = envScript;
        };

        packages.default = runScript;
        apps.default = {
          type = "app";
          program = "${runScript}/bin/metrabs-run";
        };
      }
    );
}
