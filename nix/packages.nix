# nix/packages.nix — Claudia package built with uv2nix
{ inputs, ... }: {
  perSystem = { pkgs, system, ... }:
    let
      claudiaVenv = pkgs.callPackage ./python.nix {
        inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
      };

      # Import bundled skills, excluding runtime caches
      bundledSkills = pkgs.lib.cleanSourceWith {
        src = ../skills;
        filter = path: _type:
          !(pkgs.lib.hasInfix "/index-cache/" path);
      };

      runtimeDeps = with pkgs; [
        nodejs_20 ripgrep git openssh ffmpeg
      ];

      runtimePath = pkgs.lib.makeBinPath runtimeDeps;
    in {
      packages.default = pkgs.stdenv.mkDerivation {
        pname = "claudia-autonomous";
        version = "0.1.0";

        dontUnpack = true;
        dontBuild = true;
        nativeBuildInputs = [ pkgs.makeWrapper ];

        installPhase = ''
          runHook preInstall

          mkdir -p $out/share/claudia-autonomous $out/bin
          cp -r ${bundledSkills} $out/share/claudia-autonomous/skills

          ${pkgs.lib.concatMapStringsSep "\n" (name: ''
            makeWrapper ${claudiaVenv}/bin/${name} $out/bin/${name} \
              --suffix PATH : "${runtimePath}" \
              --set CLAUDIA_BUNDLED_SKILLS $out/share/claudia-autonomous/skills
          '') [ "claudia" "claudia-autonomous" "claudia-acp" ]}

          runHook postInstall
        '';

        meta = with pkgs.lib; {
          description = "AI agent with advanced tool-calling capabilities";
          homepage = "https://github.com/NousResearch/claudia-autonomous";
          mainProgram = "claudia";
          license = licenses.mit;
          platforms = platforms.unix;
        };
      };
    };
}
