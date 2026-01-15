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
          pkgs.qt5.qtbase
          pkgs.qt5.qtwayland
          pkgs.wayland
          pkgs.xorg.libX11
          pkgs.xorg.libXext
          pkgs.xorg.libxcb
          pkgs.xorg.libXcursor
          pkgs.xorg.libXinerama
          pkgs.xorg.libXrandr
          pkgs.xorg.libXrender
          pkgs.xorg.libXfixes
          pkgs.xorg.libXi
          pkgs.xorg.libXtst
          pkgs.xorg.libXdamage
          pkgs.xorg.libSM
          pkgs.xorg.libICE
          pkgs.xorg.xcbutil
          pkgs.xorg.xcbutilimage
          pkgs.xorg.xcbutilkeysyms
          pkgs.xorg.xcbutilrenderutil
          pkgs.xorg.xcbutilwm
          pkgs.libxkbcommon
          pkgs.fontconfig
          pkgs.freetype
          pkgs.dbus
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
            export QT_QPA_PLATFORM="xcb"
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
            qt5.qtbase
            qt5.qtwayland
            wayland
            xorg.libX11
            xorg.libXext
            xorg.libxcb
            xorg.libXcursor
            xorg.libXinerama
            xorg.libXrandr
            xorg.libXrender
            xorg.libXfixes
            xorg.libXi
            xorg.libXtst
            xorg.libXdamage
            xorg.libSM
            xorg.libICE
            xorg.xcbutil
            xorg.xcbutilimage
            xorg.xcbutilkeysyms
            xorg.xcbutilrenderutil
            xorg.xcbutilwm
            libxkbcommon
            fontconfig
            freetype
            dbus
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${libraryPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            export QT_QPA_PLATFORM="xcb"
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
