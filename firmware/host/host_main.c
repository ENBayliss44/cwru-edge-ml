/* Host (macOS/Linux) harness: runs the exact firmware C on test vectors.
 *
 * usage: host_main windows.bin zq_ref.bin out.bin
 *   windows.bin : int32 n, int32 N, then n*N float32 samples (g)
 *   zq_ref.bin  : n*EM_N_FEATURES int8 (Python-quantized inputs)
 *   out.bin     : per window: float feat[F], int8 zq[F], int32 logits[C],
 *                 int32 cls, int32 logits_from_ref[C], int32 cls_from_ref
 */
#include <stdio.h>
#include <stdlib.h>

#include "../edge_ml_bench/src/edge_ml/edge_ml.h"

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s windows.bin zq_ref.bin out.bin\n", argv[0]); return 2; }
    FILE *fw = fopen(argv[1], "rb"), *fz = fopen(argv[2], "rb"), *fo = fopen(argv[3], "wb");
    if (!fw || !fz || !fo) { perror("open"); return 1; }
    int32_t n, N;
    if (fread(&n, 4, 1, fw) != 1 || fread(&N, 4, 1, fw) != 1) return 1;
    if (N != EM_WINDOW) { fprintf(stderr, "window %d != EM_WINDOW %d\n", N, EM_WINDOW); return 1; }
    static float x[EM_WINDOW];
    em_init();
    for (int w = 0; w < n; w++) {
        if (fread(x, sizeof(float), N, fw) != (size_t)N) return 1;
        int8_t zref[EM_N_FEATURES];
        if (fread(zref, 1, EM_N_FEATURES, fz) != EM_N_FEATURES) return 1;
        em_result_t r;
        em_classify(x, &r);
        int32_t lref[EM_N_CLASSES];
        int32_t cref = em_infer_q(zref, lref);
        fwrite(r.features, sizeof(float), EM_N_FEATURES, fo);
        fwrite(r.zq, 1, EM_N_FEATURES, fo);
        fwrite(r.logits, 4, EM_N_CLASSES, fo);
        fwrite(&r.cls, 4, 1, fo);
        fwrite(lref, 4, EM_N_CLASSES, fo);
        fwrite(&cref, 4, 1, fo);
    }
    fclose(fw); fclose(fz); fclose(fo);
    fprintf(stderr, "processed %d windows (static RAM %u B)\n", n, em_static_ram_bytes());
    return 0;
}
