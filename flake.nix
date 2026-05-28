{
  description = "Tiled Execution IR (TEIR) — reproducible development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;
        cppToolchain = if pkgs.stdenv.isDarwin
          then pkgs.llvmPackages_16.clang
          else pkgs.gcc12;
        commonInputs = [
          python
          python.pkgs.pip
          pkgs.uv

          cppToolchain
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config

          pkgs.openblas
          pkgs.stdenv.cc.cc.lib

          pkgs.ruff
          pkgs.clang-tools

          pkgs.git
          pkgs.gnumake
          pkgs.curl
          pkgs.cacert
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          name = "etops-teir-dev";
          packages = commonInputs;
          shellHook = ''
            export OPENBLAS_NUM_THREADS=1
            export BLAS_ROOT=${pkgs.openblas}
            export PYTHONNOUSERSITE=1
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            echo "etops-teir dev shell"
            echo "  python : $(python --version)"
            echo "  cmake  : $(cmake --version | head -1)"
            echo "  BLAS   : ${pkgs.openblas}"
            echo
            echo "Bootstrap: 'uv venv .venv && source .venv/bin/activate && uv pip install -ve \".[test]\"'"
          '';
        };

        devShells.benchmarks = pkgs.mkShell {
          name = "etops-teir-bench";
          packages = commonInputs;
          shellHook = ''
            export OPENBLAS_NUM_THREADS=1
            echo "etops-teir benchmark shell. Install competitors with:"
            echo "  uv pip install -ve .[benchmarks]"
          '';
        };
      });
}
