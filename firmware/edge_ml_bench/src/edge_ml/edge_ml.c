/* edge_ml implementation. See edge_ml.h and src/features.py / src/quantize.py. */
#include "edge_ml.h"

#include <math.h>

#define EM_PI 3.14159265358979323846
#define EM_EPS 1e-12f
#define EM_BINS_PER_BAND (EM_NFFT / 2 / EM_N_BANDS)

#if (EM_NFFT & (EM_NFFT - 1)) != 0 || EM_NFFT < EM_WINDOW
#error "EM_NFFT must be a power of two >= EM_WINDOW"
#endif

static float s_re[EM_NFFT];
static float s_im[EM_NFFT];
static float s_cos[EM_NFFT / 2];
static float s_sin[EM_NFFT / 2];

uint32_t em_static_ram_bytes(void) {
    return (uint32_t)(sizeof(s_re) + sizeof(s_im) + sizeof(s_cos) + sizeof(s_sin));
}

void em_init(void) {
    for (int k = 0; k < EM_NFFT / 2; k++) {
        double a = -2.0 * EM_PI * (double)k / (double)EM_NFFT;
        s_cos[k] = (float)cos(a);
        s_sin[k] = (float)sin(a);
    }
}

/* In-place iterative radix-2 decimation-in-time FFT on s_re/s_im. */
static void fft_inplace(void) {
    const int n = EM_NFFT;
    for (int i = 1, j = 0; i < n; i++) {           /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = s_re[i]; s_re[i] = s_re[j]; s_re[j] = t;
            t = s_im[i]; s_im[i] = s_im[j]; s_im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1, step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; k++) {
                const float wr = s_cos[k * step], wi = s_sin[k * step];
                const int a = i + k, b = a + half;
                const float xr = s_re[b] * wr - s_im[b] * wi;
                const float xi = s_re[b] * wi + s_im[b] * wr;
                s_re[b] = s_re[a] - xr; s_im[b] = s_im[a] - xi;
                s_re[a] += xr;          s_im[a] += xi;
            }
        }
    }
}

void em_features(const float *x, float *feat) {
    const int n = EM_WINDOW;
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    float s2 = 0, s3 = 0, s4 = 0, sabs = 0, mx = -INFINITY, mn = INFINITY;
    for (int i = 0; i < n; i++) {
        const float v = x[i] - mean, v2 = v * v;
        s2 += v2; s3 += v2 * v; s4 += v2 * v2; sabs += fabsf(v);
        if (v > mx) mx = v;
        if (v < mn) mn = v;
        /* Hann window into the FFT buffer while we're here */
        s_re[i] = v * (0.5f - 0.5f * cosf(2.0f * (float)EM_PI * (float)i / (float)(n - 1)));
        s_im[i] = 0.0f;
    }
    for (int i = n; i < EM_NFFT; i++) { s_re[i] = 0.0f; s_im[i] = 0.0f; }

    const float rms = sqrtf(s2 / n), mabs = sabs / n;
    const float peak = mx > -mn ? mx : -mn;
    const float r2 = rms * rms;
    feat[0] = log10f(rms + EM_EPS);
    feat[1] = log10f(mx - mn + EM_EPS);
    feat[2] = peak / rms;
    feat[3] = (s4 / n) / (r2 * r2);
    feat[4] = (s3 / n) / (r2 * rms);
    feat[5] = peak / mabs;
    feat[6] = rms / mabs;

    fft_inplace();
    for (int b = 0; b < EM_N_BANDS; b++) {
        float e = 0.0f;
        for (int k = b * EM_BINS_PER_BAND; k < (b + 1) * EM_BINS_PER_BAND; k++) {
            if (k == 0) continue;                       /* drop DC */
            e += s_re[k] * s_re[k] + s_im[k] * s_im[k];
        }
        feat[EM_N_TIME_FEATURES + b] = log10f(e + EM_EPS);
    }
}

void em_quantize_input(const float *feat, int8_t *zq) {
    for (int i = 0; i < EM_N_FEATURES; i++) {
        const float z = (feat[i] - em_mu[i]) * em_inv_sd[i];
        float q = floorf(z * em_inv_s_in + 0.5f);
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        zq[i] = (int8_t)q;
    }
}

static int32_t argmax_i32(const int32_t *v, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

int32_t em_infer_q(const int8_t *zq, int32_t *logits) {
#if EM_HAS_HIDDEN
    int8_t h[EM_N_HIDDEN];
    for (int j = 0; j < EM_N_HIDDEN; j++) {
        int32_t acc = em_b1[j];
        for (int i = 0; i < EM_N_FEATURES; i++)
            acc += (int32_t)em_W1[j * EM_N_FEATURES + i] * (int32_t)zq[i];
        int64_t r = ((int64_t)acc * EM_M1 + (1LL << (EM_S1 - 1))) >> EM_S1;
        h[j] = (int8_t)(r < 0 ? 0 : (r > 127 ? 127 : r));
    }
    for (int c = 0; c < EM_N_CLASSES; c++) {
        int32_t acc = em_b2[c];
        for (int j = 0; j < EM_N_HIDDEN; j++)
            acc += (int32_t)em_W2[c * EM_N_HIDDEN + j] * (int32_t)h[j];
        logits[c] = acc;
    }
#else
    for (int c = 0; c < EM_N_CLASSES; c++) {
        int32_t acc = em_b1[c];
        for (int i = 0; i < EM_N_FEATURES; i++)
            acc += (int32_t)em_W1[c * EM_N_FEATURES + i] * (int32_t)zq[i];
        logits[c] = acc;
    }
#endif
    return argmax_i32(logits, EM_N_CLASSES);
}

int32_t em_classify(const float *x, em_result_t *out) {
    em_features(x, out->features);
    em_quantize_input(out->features, out->zq);
    out->cls = em_infer_q(out->zq, out->logits);
    return out->cls;
}
