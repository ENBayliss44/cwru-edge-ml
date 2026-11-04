#include "lsm6dsox_fifo.h"

#define WORD_BYTES 7   /* TAG + 6 data bytes */
#define CHECK(x) do { if ((x) != 0) return LSM_ERR_BUS; } while (0)

int lsm_init(const lsm_bus_t *bus) {
    uint8_t v = 0;
    CHECK(bus->read(bus->ctx, LSM_WHO_AM_I, &v, 1));
    if (v != LSM_WHO_AM_I_VAL) return LSM_ERR_WHOAMI;

    CHECK(bus->write(bus->ctx, LSM_CTRL3_C, 0x01));            /* SW_RESET */
    int tries = 0;
    do {
        bus->delay_ms(bus->ctx, 1);
        CHECK(bus->read(bus->ctx, LSM_CTRL3_C, &v, 1));
    } while ((v & 0x01) && ++tries < 50);
    if (v & 0x01) return LSM_ERR_RESET;

    CHECK(bus->write(bus->ctx, LSM_CTRL4_C, LSM_CTRL4_C_VAL));
    CHECK(bus->write(bus->ctx, LSM_CTRL3_C, LSM_CTRL3_C_VAL));
    CHECK(bus->write(bus->ctx, LSM_CTRL2_G, 0x00));            /* gyro power-down */
    CHECK(bus->write(bus->ctx, LSM_FIFO_CTRL4, LSM_FIFO_MODE_BYPASS));  /* flush */
    CHECK(bus->write(bus->ctx, LSM_FIFO_CTRL1, 0x00));
    CHECK(bus->write(bus->ctx, LSM_FIFO_CTRL2, 0x00));
    CHECK(bus->write(bus->ctx, LSM_FIFO_CTRL3, LSM_FIFO_CTRL3_VAL));
    CHECK(bus->write(bus->ctx, LSM_CTRL1_XL, LSM_CTRL1_XL_VAL));
    CHECK(bus->write(bus->ctx, LSM_FIFO_CTRL4, LSM_FIFO_MODE_CONTINUOUS));
    return LSM_OK;
}

int lsm_freq_fine(const lsm_bus_t *bus, int8_t *value) {
    uint8_t v;
    CHECK(bus->read(bus->ctx, LSM_FREQ_FINE, &v, 1));
    *value = (int8_t)v;
    return LSM_OK;
}

int lsm_fifo_drain(const lsm_bus_t *bus, int axis, float *out, int max_samples,
                   lsm_stats_t *st) {
    uint8_t s[2];
    if (bus->read(bus->ctx, LSM_FIFO_STATUS1, s, 2) != 0) { st->bus_errors++; return LSM_ERR_BUS; }
    if (s[1] & 0x48) st->overruns++;                     /* FIFO_OVR_IA | FIFO_OVR_LATCHED */
    int pending = (int)s[0] | ((int)(s[1] & 0x03) << 8);

    int n = 0;
    for (int w = 0; w < pending && n < max_samples; w++) {
        uint8_t b[WORD_BYTES];
        if (bus->read(bus->ctx, LSM_FIFO_DATA_TAG, b, WORD_BYTES) != 0) {
            st->bus_errors++;
            return n > 0 ? n : LSM_ERR_BUS;
        }
        const uint8_t tag = b[0] >> 3, cnt = (b[0] >> 1) & 0x03;
        if (tag != LSM_TAG_XL_NC) { st->other_words++; continue; }
        if (st->have_last && cnt != ((st->last_cnt + 1) & 0x03)) st->tag_gaps++;
        st->last_cnt = cnt;
        st->have_last = 1;
        const int16_t raw = (int16_t)((uint16_t)b[1 + 2 * axis] |
                                      ((uint16_t)b[2 + 2 * axis] << 8));
        out[n++] = (float)raw * LSM_G_PER_LSB_4G;
        st->xl_samples++;
    }
    return n;
}
