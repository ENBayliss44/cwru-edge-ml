#include "window_pp.h"

void window_pp_init(window_pp_t *pp, float *a, float *b, int n) {
    pp->buf[0] = a; pp->buf[1] = b; pp->n = n;
    pp->fill = 0; pp->pos = 0;
    pp->busy[0] = pp->busy[1] = 0;
    pp->completed = pp->dropped = 0;
}

int window_pp_push(window_pp_t *pp, const float *s, int k, int *ready_idx) {
    int done = 0;
    for (int i = 0; i < k; i++) {
        pp->buf[pp->fill][pp->pos++] = s[i];
        if (pp->pos < pp->n) continue;
        pp->pos = 0;
        const int next = pp->fill ^ 1;
        if (pp->busy[next]) {          /* consumer still on the other buffer */
            pp->dropped++;             /* discard this window, refill same buffer */
            continue;
        }
        pp->busy[pp->fill] = 1;
        *ready_idx = pp->fill;
        pp->completed++;
        done++;
        pp->fill = next;
    }
    return done;
}

void window_pp_release(window_pp_t *pp, int idx) {
    pp->busy[idx] = 0;
}
