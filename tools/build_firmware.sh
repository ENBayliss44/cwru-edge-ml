#!/bin/sh
# Compile the ESP32 firmware (full + ML-stubbed baseline) for both model rates
# and record flash / static-RAM deltas in results/firmware_size.json.
# Requires arduino-cli with the esp32:esp32 core.
# ctags is pointed at a no-op: arduino-cli's bundled ctags is x86-only, and the
# sketch defines every function before use, so no prototypes are needed.
set -e
cd "$(dirname "$0")/.."
FQBN=esp32:esp32:adafruit_feather_esp32_v2
OUT=results/firmware_size.json
echo "{" > $OUT
first=1
for model in 6664_mlp_load3 1666_mlp_load3; do
  for mode in full baseline; do
    D="-DEM_MODEL_HEADER=\"models/model_${model}.h\""
    [ $mode = baseline ] && D="$D -DEM_BASELINE"
    log=build/fw_${model}_${mode}.log
    arduino-cli compile --fqbn $FQBN --warnings all --build-path build/fw_${model}_${mode} \
      --build-property "tools.ctags.pattern=/usr/bin/true" \
      --build-property "compiler.cpp.extra_flags=$D" \
      --build-property "compiler.c.extra_flags=$D" firmware/edge_ml_bench > $log 2>&1
    eval "flash_${mode}=$(sed -n 's/^Sketch uses \([0-9]*\).*/\1/p' $log)"
    eval "ram_${mode}=$(sed -n 's/^Global variables use \([0-9]*\).*/\1/p' $log)"
  done
  [ $first = 1 ] || echo "," >> $OUT; first=0
  printf ' "%s": {"flash_full": %s, "flash_baseline": %s, "flash_edge_ml": %s, "static_ram_full": %s, "static_ram_baseline": %s, "static_ram_edge_ml": %s}' \
    $model $flash_full $flash_baseline $((flash_full - flash_baseline)) \
    $ram_full $ram_baseline $((ram_full - ram_baseline)) >> $OUT
  echo "$model: edge_ml flash +$((flash_full - flash_baseline)) B, static RAM +$((ram_full - ram_baseline)) B"
done
echo "" >> $OUT; echo "}" >> $OUT
