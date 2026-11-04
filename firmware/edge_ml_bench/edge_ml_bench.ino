// edge_ml on the Adafruit ESP32 Feather V2.
//
// Two input paths into the same classifier:
//   1. Streamed: host sends held-out CWRU windows over serial (benchmark and
//      Python cross-check, see tools/stream_bench.py).
//   2. Live: LSM6DSOX over SPI at 6667 Hz through its FIFO (see
//      src/sensor/). A FreeRTOS task on core 0 drains the FIFO into ping-pong
//      window buffers; loop() on core 1 classifies completed windows.
//
// Serial protocol (921600 baud; binary in, text lines out):
//   'W' + EM_WINDOW float32 LE  -> R,<cls>,<l0..l3>,<t_feat_us>,<t_infer_us>,<t_total_us>
//   'I'                         -> I,<fs>,<window>,<nfft>,<static_ram>,<free_heap>,
//                                    <min_free_heap>,<stack_hwm>,<cpu_mhz>
//   'S'  sensor self-test (EM_SELFTEST_S seconds)
//        -> S,<measured_hz>,<expected_hz>,<freq_fine>,<samples>,<overruns>,<tag_gaps>,
//             <other_words>,<bus_errors>
//   'L'  live classification until any byte is received; one line per window
//        -> L,<cls>,<l0..l3>,<t_total_us>,<rms_g>,<window_hz>,<overruns>,<tag_gaps>,<dropped>
//   errors -> E,<reason>
// Build with -DEM_BASELINE to stub out edge_ml (for flash/RAM deltas).
#include <Arduino.h>
#include <SPI.h>
#include "esp_timer.h"
#include "src/edge_ml/edge_ml.h"
#include "src/sensor/lsm6dsox_fifo.h"
#include "src/sensor/sensor_config.h"
#include "src/sensor/window_pp.h"

#ifdef EM_BASELINE
// No ML calls: --gc-sections drops edge_ml, so (full - baseline) isolates it.
#define EM_INIT()
#define EM_RAM() 0u
#else
#define EM_INIT() em_init()
#define EM_RAM() em_static_ram_bytes()
#endif

// ---------------------------------------------------------------- buffers
static float *g_win[2];            // heap: 2 x EM_WINDOW floats (also the 'W' buffer)
static window_pp_t g_pp;
static QueueHandle_t g_ready_q;    // completed window indices, sampler -> loop
static TaskHandle_t g_sampler;
static volatile bool g_sampling = false;
static volatile bool g_to_window = false;   // false: count only (self-test)
static lsm_stats_t g_st;
static volatile int64_t g_win_t[2];         // esp_timer at window completion

// ---------------------------------------------------------------- SPI bus
static SPIClass g_spi(HSPI);
static const SPISettings g_spi_cfg(EM_SPI_HZ, MSBFIRST, SPI_MODE3);

static int spi_read(void *, uint8_t reg, uint8_t *buf, uint16_t len) {
  g_spi.beginTransaction(g_spi_cfg);
  digitalWrite(EM_SPI_CS, LOW);
  g_spi.transfer(reg | 0x80);
  memset(buf, 0, len);
  g_spi.transfer(buf, len);
  digitalWrite(EM_SPI_CS, HIGH);
  g_spi.endTransaction();
  return 0;
}

static int spi_write(void *, uint8_t reg, uint8_t val) {
  g_spi.beginTransaction(g_spi_cfg);
  digitalWrite(EM_SPI_CS, LOW);
  g_spi.transfer(reg & 0x7F);
  g_spi.transfer(val);
  digitalWrite(EM_SPI_CS, HIGH);
  g_spi.endTransaction();
  return 0;
}

static void bus_delay(void *, uint32_t ms) { delay(ms); }
static const lsm_bus_t g_bus = {spi_read, spi_write, bus_delay, nullptr};

// ---------------------------------------------------------------- sampler (core 0)
static void sampler_task(void *) {
  static float chunk[256];
  uint32_t settle = EM_SETTLE_SAMPLES;
  while (g_sampling) {
    int k;
    do {
      k = lsm_fifo_drain(&g_bus, EM_SENSOR_AXIS, chunk, 256, &g_st);
      if (k <= 0) break;
      int off = 0;
      if (settle) {                       // drop start-up transient
        off = (uint32_t)k < settle ? k : (int)settle;
        settle -= off;
      }
      int idx;
      if (g_to_window && k > off && window_pp_push(&g_pp, chunk + off, k - off, &idx)) {
        g_win_t[idx] = esp_timer_get_time();
        xQueueSend(g_ready_q, &idx, 0);
      }
    } while (k == 256);
    vTaskDelay(pdMS_TO_TICKS(EM_POLL_MS));
  }
  g_sampler = nullptr;
  vTaskDelete(nullptr);
}

static bool sensor_start(bool to_window) {
  if (g_sampling) return true;
  int rc = lsm_init(&g_bus);
  if (rc != LSM_OK) {
    Serial.printf("E,lsm_init,%d\n", rc);
    return false;
  }
  memset(&g_st, 0, sizeof(g_st));
  window_pp_init(&g_pp, g_win[0], g_win[1], EM_WINDOW);
  xQueueReset(g_ready_q);
  g_to_window = to_window;
  g_sampling = true;
  xTaskCreatePinnedToCore(sampler_task, "lsm_fifo", 4096, nullptr, 5, &g_sampler, 0);
  return true;
}

static void sensor_stop() {
  g_sampling = false;
  while (g_sampler) delay(1);
  spi_write(nullptr, LSM_FIFO_CTRL4, LSM_FIFO_MODE_BYPASS);
  spi_write(nullptr, LSM_CTRL1_XL, 0x00);   // accelerometer power-down
}

// ---------------------------------------------------------------- commands
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

static void cmd_window() {
  if (!read_exact((uint8_t *)g_win[0], EM_WINDOW * sizeof(float), 2000)) {
    Serial.println("E,timeout");
    return;
  }
#ifdef EM_BASELINE
  volatile float sink = g_win[0][0];
  (void)sink;
  Serial.println("R,0,0,0,0,0,0,0,0");
#else
  em_result_t r;
  int64_t t0 = esp_timer_get_time();
  em_features(g_win[0], r.features);
  int64_t t1 = esp_timer_get_time();
  em_quantize_input(r.features, r.zq);
  r.cls = em_infer_q(r.zq, r.logits);
  int64_t t2 = esp_timer_get_time();
  Serial.printf("R,%d,%d,%d,%d,%d,%lld,%lld,%lld\n", (int)r.cls, (int)r.logits[0],
                (int)r.logits[1], (int)r.logits[2], (int)r.logits[3],
                (long long)(t1 - t0), (long long)(t2 - t1), (long long)(t2 - t0));
#endif
}

static void cmd_selftest() {
  if (!sensor_start(false)) return;
  delay(1000);                                    // past settle + FIFO start
  uint32_t n0 = g_st.xl_samples;
  int64_t t0 = esp_timer_get_time();
  delay(EM_SELFTEST_S * 1000);
  uint32_t n1 = g_st.xl_samples;
  int64_t t1 = esp_timer_get_time();
  sensor_stop();
  int8_t ff = 0;
  lsm_freq_fine(&g_bus, &ff);
  const double measured = (double)(n1 - n0) * 1e6 / (double)(t1 - t0);
  const double expected = LSM_NOMINAL_ODR_HZ * (1.0 + 0.0015 * ff);
  Serial.printf("S,%.2f,%.2f,%d,%u,%u,%u,%u,%u\n", measured, expected, (int)ff,
                (unsigned)(n1 - n0), (unsigned)g_st.overruns, (unsigned)g_st.tag_gaps,
                (unsigned)g_st.other_words, (unsigned)g_st.bus_errors);
}

static void cmd_live() {
#if EM_FS != 6664
  Serial.println("E,live mode needs a 6664 Hz model");
  return;
#else
  if (!sensor_start(true)) return;
  int64_t prev_t = 0;
  while (!Serial.available()) {
    int idx;
    if (xQueueReceive(g_ready_q, &idx, pdMS_TO_TICKS(20)) != pdTRUE) continue;
    const int64_t t_done = g_win_t[idx];
    const double hz = prev_t ? EM_WINDOW * 1e6 / (double)(t_done - prev_t) : 0.0;
    prev_t = t_done;
#ifdef EM_BASELINE
    window_pp_release(&g_pp, idx);
    Serial.printf("L,0,0,0,0,0,0,0,%.2f,%u,%u,%u\n", hz, (unsigned)g_st.overruns,
                  (unsigned)g_st.tag_gaps, (unsigned)g_pp.dropped);
#else
    em_result_t r;
    int64_t t0 = esp_timer_get_time();
    em_classify(g_win[idx], &r);
    int64_t t1 = esp_timer_get_time();
    window_pp_release(&g_pp, idx);
    Serial.printf("L,%d,%d,%d,%d,%d,%lld,%.5f,%.2f,%u,%u,%u\n", (int)r.cls,
                  (int)r.logits[0], (int)r.logits[1], (int)r.logits[2], (int)r.logits[3],
                  (long long)(t1 - t0), powf(10.0f, r.features[0]), hz,
                  (unsigned)g_st.overruns, (unsigned)g_st.tag_gaps, (unsigned)g_pp.dropped);
#endif
  }
  while (Serial.available()) Serial.read();
  sensor_stop();
#endif
}

// ---------------------------------------------------------------- Arduino
void setup() {
  Serial.setRxBufferSize(8192);
  Serial.begin(921600);
  g_win[0] = (float *)malloc(EM_WINDOW * sizeof(float));
  g_win[1] = (float *)malloc(EM_WINDOW * sizeof(float));
  g_ready_q = xQueueCreate(2, sizeof(int));
  pinMode(EM_SPI_CS, OUTPUT);
  digitalWrite(EM_SPI_CS, HIGH);              // CS high before the bus starts
  g_spi.begin(EM_SPI_SCK, EM_SPI_MISO, EM_SPI_MOSI, -1);
  EM_INIT();
  delay(200);
  if (!g_win[0] || !g_win[1] || !g_ready_q) Serial.println("E,alloc");
  report_info();
}

void loop() {
  if (!Serial.available()) return;
  switch (Serial.read()) {
    case 'I': report_info(); break;
    case 'W': cmd_window(); break;
    case 'S': cmd_selftest(); break;
    case 'L': cmd_live(); break;
    default: break;
  }
}
