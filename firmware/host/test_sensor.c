/* Host tests for the LSM6DSOX FIFO driver and ping-pong window assembler,
 * against a simulated sensor (register file + 6667 Hz FIFO).
 * Build/run: see tools/run_host_tests.sh. Exit code 0 = all pass. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../edge_ml_bench/src/edge_ml/edge_ml.h"
#include "../edge_ml_bench/src/sensor/lsm6dsox_fifo.h"
#include "../edge_ml_bench/src/sensor/window_pp.h"

/* ---------------- simulated sensor ---------------- */
#define FIFO_CAP 512           /* conservative depth; real part holds more */
#define SIM_FS 6667.0
typedef struct {
    uint8_t reg[256];
    uint8_t whoami;
    int reset_reads;
    uint8_t fifo[FIFO_CAP][7];
    int head, count;
    double t, next_sample_t;   /* simulated time (s) */
    uint32_t sample_idx;
    uint8_t cnt;
    int ovr_latched;
    int inject_gap_at;         /* sample index to skip a TAG_CNT step, -1 = none */
    double amp[3], freq;
    int write_log[64][2], n_writes;
} sim_t;

static void sim_produce_until(sim_t *s, double t) {
    while (s->next_sample_t <= t) {
        const int running = s->reg[LSM_FIFO_CTRL4] == LSM_FIFO_MODE_CONTINUOUS &&
                            (s->reg[LSM_FIFO_CTRL3] & 0x0F) == LSM_ODR_6667;
        if (running) {
            if ((int)s->sample_idx == s->inject_gap_at) s->cnt = (s->cnt + 1) & 3;
            uint8_t *w;
            if (s->count == FIFO_CAP) {                 /* continuous: overwrite oldest */
                s->head = (s->head + 1) % FIFO_CAP; s->count--; s->ovr_latched = 1;
            }
            w = s->fifo[(s->head + s->count) % FIFO_CAP];
            w[0] = (uint8_t)((LSM_TAG_XL_NC << 3) | (s->cnt << 1));
            for (int a = 0; a < 3; a++) {
                double g = s->amp[a] * sin(2 * M_PI * s->freq * s->next_sample_t) + (a == 2 ? 1.0 : 0.0);
                long raw = lround(g / LSM_G_PER_LSB_4G);
                if (raw > 32767) raw = 32767;
                if (raw < -32768) raw = -32768;
                int16_t r = (int16_t)raw;
                w[1 + 2 * a] = (uint8_t)(r & 0xFF); w[2 + 2 * a] = (uint8_t)((uint16_t)r >> 8);
            }
            s->count++;
            s->cnt = (s->cnt + 1) & 3;
            s->sample_idx++;
        }
        s->next_sample_t += 1.0 / SIM_FS;
    }
}

static int sim_read(void *ctx, uint8_t r, uint8_t *buf, uint16_t len) {
    sim_t *s = ctx;
    sim_produce_until(s, s->t);
    if (r == LSM_WHO_AM_I && len == 1) { buf[0] = s->whoami; return 0; }
    if (r == LSM_CTRL3_C && len == 1) {
        if ((s->reg[r] & 1) && ++s->reset_reads >= 2) s->reg[r] &= ~1;
        buf[0] = s->reg[r]; return 0;
    }
    if (r == LSM_FIFO_STATUS1 && len == 2) {
        buf[0] = (uint8_t)(s->count & 0xFF);
        buf[1] = (uint8_t)(((s->count >> 8) & 0x03) | (s->ovr_latched ? 0x48 : 0));
        s->ovr_latched = 0;                       /* cleared on read */
        return 0;
    }
    if (r == LSM_FIFO_DATA_TAG && len == 7) {
        if (s->count == 0) return -1;
        memcpy(buf, s->fifo[s->head], 7);
        s->head = (s->head + 1) % FIFO_CAP; s->count--;
        return 0;
    }
    for (int i = 0; i < len; i++) buf[i] = s->reg[(uint8_t)(r + i)];
    return 0;
}
static int sim_write(void *ctx, uint8_t r, uint8_t v) {
    sim_t *s = ctx;
    s->reg[r] = v;
    if (s->n_writes < 64) { s->write_log[s->n_writes][0] = r; s->write_log[s->n_writes][1] = v; s->n_writes++; }
    if (r == LSM_CTRL3_C && (v & 1)) s->reset_reads = 0;
    if (r == LSM_FIFO_CTRL4 && v == LSM_FIFO_MODE_BYPASS) { s->count = 0; s->head = 0; }
    if (r == LSM_FIFO_CTRL4 && v == LSM_FIFO_MODE_CONTINUOUS) s->next_sample_t = s->t;
    return 0;
}
static void sim_delay(void *ctx, uint32_t ms) { ((sim_t *)ctx)->t += ms / 1000.0; }

static sim_t sim;
static lsm_bus_t bus = {sim_read, sim_write, sim_delay, &sim};

static void sim_reset(void) {
    memset(&sim, 0, sizeof sim);
    sim.whoami = LSM_WHO_AM_I_VAL; sim.reg[LSM_CTRL3_C] = 0x04;
    sim.inject_gap_at = -1; sim.amp[0] = 0.1; sim.amp[1] = 0.2; sim.amp[2] = 0.5; sim.freq = 160.0;
}

/* ---------------- tests ---------------- */
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int final_write(int r) {
    for (int i = sim.n_writes - 1; i >= 0; i--) if (sim.write_log[i][0] == r) return sim.write_log[i][1];
    return -1;
}

/* Poll the FIFO every poll_ms for seconds; returns samples collected. */
static float collected[2 * 6667 + 16];
static int run(double seconds, double poll_ms, int axis, lsm_stats_t *st, int pause_at_ms, int pause_ms) {
    int n = 0;
    for (double ms = 0; ms < seconds * 1000.0; ms += poll_ms) {
        if (pause_at_ms >= 0 && (int)ms == pause_at_ms) { sim.t += pause_ms / 1000.0; }
        sim.t += poll_ms / 1000.0;
        int cap = (int)(sizeof collected / sizeof collected[0]) - n;
        int k = lsm_fifo_drain(&bus, axis, collected + n, cap > 1024 ? 1024 : cap, st);
        if (k > 0) n += k;
    }
    return n;
}

static void test_init(void) {
    sim_reset();
    EXPECT(lsm_init(&bus) == LSM_OK, "init failed");
    EXPECT(final_write(LSM_CTRL1_XL) == 0xA8, "CTRL1_XL=0x%02X want 0xA8 (6667 Hz, +/-4 g)", final_write(LSM_CTRL1_XL));
    EXPECT(final_write(LSM_CTRL2_G) == 0x00, "gyro not powered down");
    EXPECT(final_write(LSM_CTRL3_C) == 0x44, "CTRL3_C=0x%02X want 0x44", final_write(LSM_CTRL3_C));
    EXPECT(final_write(LSM_CTRL4_C) == 0x04, "I2C not disabled");
    EXPECT(final_write(LSM_FIFO_CTRL3) == 0x0A, "FIFO_CTRL3=0x%02X want 0x0A", final_write(LSM_FIFO_CTRL3));
    EXPECT(sim.write_log[0][0] == LSM_CTRL3_C && sim.write_log[0][1] == 0x01, "first write must be SW_RESET");
    EXPECT(sim.write_log[sim.n_writes - 1][0] == LSM_FIFO_CTRL4 &&
           sim.write_log[sim.n_writes - 1][1] == LSM_FIFO_MODE_CONTINUOUS, "FIFO must be started last");

    sim_reset(); sim.whoami = 0x6A;
    EXPECT(lsm_init(&bus) == LSM_ERR_WHOAMI, "wrong WHO_AM_I accepted");
}

static void test_rate_units_axes(void) {
    for (int axis = 0; axis < 3; axis++) {
        sim_reset(); lsm_init(&bus);
        lsm_stats_t st = {0};
        int n = run(1.0, 2.0, axis, &st, -1, 0);
        EXPECT(n >= 6660 && n <= 6672, "axis %d: %d samples in 1 s, want ~6667", axis, n);
        EXPECT(st.tag_gaps == 0 && st.overruns == 0 && st.other_words == 0 && st.bus_errors == 0,
               "axis %d: gaps=%u ovr=%u other=%u err=%u", axis, st.tag_gaps, st.overruns, st.other_words, st.bus_errors);
        double mean = 0, ss = 0;
        for (int i = 0; i < 6664; i++) mean += collected[i];
        mean /= 6664;
        for (int i = 0; i < 6664; i++) ss += (collected[i] - mean) * (collected[i] - mean);
        double rms = sqrt(ss / 6664), want = sim.amp[axis] / sqrt(2.0);
        EXPECT(fabs(rms - want) < 1e-3, "axis %d: AC rms %.5f g, want %.5f g", axis, rms, want);
        EXPECT(fabs(mean - (axis == 2 ? 1.0 : 0.0)) < 2e-3, "axis %d: mean %.4f g", axis, mean);
    }
}

static void test_gap_and_overrun(void) {
    sim_reset(); lsm_init(&bus); sim.inject_gap_at = 3000;
    lsm_stats_t st = {0};
    run(1.0, 2.0, 2, &st, -1, 0);
    EXPECT(st.tag_gaps == 1, "injected TAG_CNT gap: detected %u, want 1", st.tag_gaps);

    sim_reset(); lsm_init(&bus);
    memset(&st, 0, sizeof st);
    run(1.0, 2.0, 2, &st, 400, 200);   /* stall 200 ms: 1333 samples > FIFO depth */
    EXPECT(st.overruns >= 1, "200 ms stall not reported as overrun");
}

static void test_pingpong(void) {
    static float a[100], b[100];
    window_pp_t pp; window_pp_init(&pp, a, b, 100);
    float s[30]; for (int i = 0; i < 30; i++) s[i] = (float)i;
    int idx = -1, got = 0;
    for (int i = 0; i < 4; i++) got += window_pp_push(&pp, s, 30, &idx);   /* 120 samples */
    EXPECT(got == 1 && idx == 0 && pp.busy[0], "first window not handed off");
    EXPECT(a[99] == s[9] && a[0] == 0.0f, "window contents wrong");
    /* Slow consumer: B fills while A is still held -> B dropped, refilled. */
    idx = -1; got = 0;
    for (int i = 0; i < 3; i++) got += window_pp_push(&pp, s, 30, &idx);   /* B reaches 110 -> full at 100 */
    EXPECT(got == 0 && pp.dropped == 1 && !pp.busy[1], "B must be dropped while A is held (got %d, dropped %u)", got, pp.dropped);
    EXPECT(idx == -1, "dropped window must not be handed off");
    /* Timely consumer: release A before B completes -> B handed off, no drop. */
    window_pp_release(&pp, 0);
    got = 0;
    for (int i = 0; i < 3; i++) got += window_pp_push(&pp, s, 30, &idx);   /* B: 10 + 90 = 100 */
    EXPECT(got == 1 && idx == 1 && pp.busy[1] && pp.dropped == 1, "B not handed off after A released");
    EXPECT(pp.completed == 2, "completed=%u want 2", pp.completed);
}

static void test_end_to_end_features(void) {
    /* Sensor -> FIFO driver -> ping-pong -> edge_ml features (6664 Hz model). */
    sim_reset(); lsm_init(&bus);
    static float A[EM_WINDOW], B[EM_WINDOW];
    window_pp_t pp; window_pp_init(&pp, A, B, EM_WINDOW);
    lsm_stats_t st = {0};
    float chunk[256]; int idx = -1, ready = 0;
    for (int ms = 0; ms < 1200 && !ready; ms += 2) {
        sim.t += 0.002;
        int k = lsm_fifo_drain(&bus, 2, chunk, 256, &st);
        if (k > 0) ready = window_pp_push(&pp, chunk, k, &idx);
    }
    EXPECT(ready && idx == 0, "no window assembled in 1.2 s");
    em_init();
    float f[EM_N_FEATURES];
    em_features(pp.buf[idx], f);
    double rms = pow(10.0, f[0]), want = sim.amp[2] / sqrt(2.0);
    EXPECT(fabs(rms - want) / want < 2e-3, "edge_ml RMS feature %.5f g, want %.5f g", rms, want);
    int8_t zq[EM_N_FEATURES]; int32_t lg[EM_N_CLASSES];
    em_quantize_input(f, zq);
    int c = em_infer_q(zq, lg);
    EXPECT(c >= 0 && c < EM_N_CLASSES, "classifier returned %d", c);
}

int main(void) {
    test_init();
    test_rate_units_axes();
    test_gap_and_overrun();
    test_pingpong();
    test_end_to_end_features();
    printf(fails ? "%d FAILED\n" : "all sensor tests passed\n", fails);
    return fails != 0;
}
