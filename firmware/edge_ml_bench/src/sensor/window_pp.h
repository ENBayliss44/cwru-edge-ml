/* Ping-pong window assembler: the sampler fills one buffer while the
 * classifier reads the other. If the classifier has not released the next
 * buffer when the current one fills, the just-completed window is dropped
 * (counted) and refilled, so samples are never written into a buffer that is
 * being classified. Single producer, single consumer; the caller provides the
 * cross-core handoff (FreeRTOS queue on the ESP32). */
#ifndef WINDOW_PP_H
#define WINDOW_PP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float   *buf[2];
    int      n;            /* window length (samples) */
    int      fill;         /* buffer being filled */
    int      pos;
    volatile uint8_t busy[2];   /* 1 = handed to consumer, not yet released */
    uint32_t completed, dropped;
} window_pp_t;

void window_pp_init(window_pp_t *pp, float *a, float *b, int n);

/* Append k samples. Writes the index of a completed window to *ready_idx and
 * returns how many windows completed (0 or 1 per call while k <= n). */
int window_pp_push(window_pp_t *pp, const float *s, int k, int *ready_idx);

void window_pp_release(window_pp_t *pp, int idx);

#ifdef __cplusplus
}
#endif
#endif
