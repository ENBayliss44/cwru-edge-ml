// ESP32 on-device benchmark for edge_ml (Adafruit ESP32 Feather V2).
//
// Host streams held-out CWRU windows over serial; the board runs the exact
// C pipeline (features + int8 inference) and reports class, int32 logits and
// esp_timer latencies. See tools/stream_bench.py.
//
// Protocol (binary in, text out, 921600 baud):
//   host -> 'W' + EM_WINDOW little-endian float32 samples (g)
//   dev  -> "R,<cls>,<l0>,<l1>,<l2>,<l3>,<t_feat_us>,<t_infer_us>,<t_total_us>\n"
//   host -> 'I'  : dev -> "I,<fs>,<window>,<nfft>,<static_ram>,<free_heap>,
//                          <min_free_heap>,<stack_hwm_bytes>,<cpu_mhz>\n"
// Build with -DEM_BASELINE to stub out edge_ml (for flash/RAM deltas).
#include <Arduino.h>
#include "esp_timer.h"
#include "src/edge_ml/edge_ml.h"

static float x_buf[EM_WINDOW];

#ifdef EM_BASELINE
// Same I/O, no ML calls: the linker's --gc-sections drops edge_ml entirely,
// so (full build - baseline build) isolates edge_ml's flash and RAM.
#define EM_INIT()
#define EM_RAM() 0u
#else
#define EM_INIT() em_init()
#define EM_RAM() em_static_ram_bytes()
#endif

static bool read_exact(uint8_t *dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n) {
    int a = Serial.available();
    if (a > 0) {
      got += Serial.readBytes(dst + got, min((size_t)a, n - got));
      t0 = millis();
    } else if (millis() - t0 > timeout_ms) {
      return false;
    }
  }
  return true;
}

static void report_info() {
  Serial.printf("I,%d,%d,%d,%u,%u,%u,%u,%u\n", EM_FS, EM_WINDOW, EM_NFFT,
                (unsigned)EM_RAM(), (unsigned)ESP.getFreeHeap(),
                (unsigned)ESP.getMinFreeHeap(),
                (unsigned)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)),
                (unsigned)getCpuFrequencyMhz());
}

void setup() {
  Serial.setRxBufferSize(8192);
  Serial.begin(921600);
  EM_INIT();
  delay(200);
  report_info();
}

void loop() {
  if (!Serial.available()) return;
  int cmd = Serial.read();
  if (cmd == 'I') {
    report_info();
  } else if (cmd == 'W') {
    if (!read_exact((uint8_t *)x_buf, sizeof(x_buf), 2000)) {
      Serial.println("E,timeout");
      return;
    }
#ifdef EM_BASELINE
    volatile float sink = x_buf[0];
    (void)sink;
    Serial.println("R,0,0,0,0,0,0,0,0");
#else
    em_result_t r;
    int64_t t0 = esp_timer_get_time();
    em_features(x_buf, r.features);
    int64_t t1 = esp_timer_get_time();
    em_quantize_input(r.features, r.zq);
    r.cls = em_infer_q(r.zq, r.logits);
    int64_t t2 = esp_timer_get_time();
    Serial.printf("R,%d,%d,%d,%d,%d,%lld,%lld,%lld\n", (int)r.cls,
                  (int)r.logits[0], (int)r.logits[1], (int)r.logits[2], (int)r.logits[3],
                  (long long)(t1 - t0), (long long)(t2 - t1), (long long)(t2 - t0));
#endif
  }
}
