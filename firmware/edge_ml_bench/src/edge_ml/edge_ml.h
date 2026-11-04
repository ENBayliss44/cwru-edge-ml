/* edge_ml: bearing-fault features + int8 classifier in plain C99.
 *
 * No heap, no external dependencies beyond <math.h>. All buffers are static
 * and sized at compile time from the selected model header (EM_WINDOW,
 * EM_NFFT). Mirrors src/features.py and src/quantize.py in the Python repo.
 */
#ifndef EDGE_ML_H
#define EDGE_ML_H

#include <stdint.h>

#ifndef EM_MODEL_HEADER
#include "model_config.h"   /* defines EM_MODEL_HEADER */
#endif
#include EM_MODEL_HEADER

#ifdef __cplusplus
extern "C" {
#endif

#define EM_N_TIME_FEATURES 7
#define EM_N_BANDS 16

typedef struct {
    float   features[EM_N_FEATURES];
    int8_t  zq[EM_N_FEATURES];
    int32_t logits[EM_N_CLASSES];
    int32_t cls;               /* 0 normal, 1 inner, 2 outer, 3 ball */
} em_result_t;

/* Build twiddle tables. Call once before anything else. */
void em_init(void);

/* Features from one window of EM_WINDOW samples (acceleration in g). */
void em_features(const float *x, float *feat);

/* Standardize + quantize features to int8 model input. */
void em_quantize_input(const float *feat, int8_t *zq);

/* Integer-only inference: int8 input -> int32 logits, returns argmax. */
int32_t em_infer_q(const int8_t *zq, int32_t *logits);

/* Full pipeline: window -> class. */
int32_t em_classify(const float *x, em_result_t *out);

/* Static RAM used by edge_ml buffers (bytes), for reporting. */
uint32_t em_static_ram_bytes(void);

#ifdef __cplusplus
}
#endif
#endif
