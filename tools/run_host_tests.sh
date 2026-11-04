#!/bin/sh
# Build and run host-side C tests (no hardware needed).
set -e
cd "$(dirname "$0")/.."
SRC=firmware/edge_ml_bench/src
clang -std=c99 -O2 -Wall -Wextra -Werror \
  -DEM_MODEL_HEADER='"models/model_6664_mlp_load3.h"' -I $SRC/edge_ml \
  firmware/host/test_sensor.c $SRC/sensor/lsm6dsox_fifo.c $SRC/sensor/window_pp.c \
  $SRC/edge_ml/edge_ml.c -lm -o build/test_sensor
./build/test_sensor
