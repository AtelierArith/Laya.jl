#!/usr/bin/env bash
# Build mlx-c (submodule deps/mlx-c) against the MLX of the extern/laya-mlx venv and
# install it into deps/usr, where LayaMLX looks for libmlxc.dylib.
#
#   git submodule update --init deps/mlx-c
#   deps/build.sh [prefix]            (default prefix: deps/usr)
set -euo pipefail
cd "$(dirname "$0")"
prefix=${1:-$PWD/usr}
python=../extern/laya-mlx/.venv/bin/python
mlx=$("$python" -c 'import mlx; print(list(mlx.__path__)[0])')
echo "MLX $("$python" -c 'import mlx.core as mx; print(mx.__version__)') at $mlx"
cmake -S mlx-c -B mlx-c/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_USE_SYSTEM_MLX=ON -DMLX_C_BUILD_EXAMPLES=OFF \
  -DMLX_DIR="$mlx/share/cmake/MLX" -DCMAKE_INSTALL_RPATH="$mlx/lib" \
  -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build mlx-c/build -j
cmake --install mlx-c/build
