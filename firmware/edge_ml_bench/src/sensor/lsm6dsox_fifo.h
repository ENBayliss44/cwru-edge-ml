/* Register-level LSM6DSOX driver: accelerometer at 6667 Hz through the FIFO.
 *
 * Plain C99. The bus is abstracted behind callbacks so the same code runs on
 * the ESP32 (SPI) and against a simulated sensor in firmware/host/.
 * Register addresses and bit fields: ST LSM6DSOX datasheet (DS12814), sections
 * "Register mapping" and "FIFO". Only the accelerometer is batched; gyro off.
 */
#ifndef LSM6DSOX_FIFO_H
#define LSM6DSOX_FIFO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Registers */
#define LSM_FIFO_CTRL1     0x07
#define LSM_FIFO_CTRL2     0x08
#define LSM_FIFO_CTRL3     0x09   /* BDR_GY[7:4] | BDR_XL[3:0] */
#define LSM_FIFO_CTRL4     0x0A   /* FIFO_MODE[2:0] */
#define LSM_WHO_AM_I       0x0F
#define LSM_CTRL1_XL       0x10   /* ODR_XL[7:4] | FS_XL[3:2] | LPF2_XL_EN[1] */
#define LSM_CTRL2_G        0x11
#define LSM_CTRL3_C        0x12   /* BOOT | BDU | H_LACTIVE | PP_OD | SIM | IF_INC | - | SW_RESET */
#define LSM_CTRL4_C        0x13   /* bit 2: I2C_disable */
#define LSM_FIFO_STATUS1   0x3A   /* DIFF_FIFO[7:0] */
#define LSM_FIFO_STATUS2   0x3B   /* WTM_IA | OVR_IA | FULL_IA | CNT_BDR_IA | OVR_LATCHED | - | DIFF_FIFO[9:8] */
#define LSM_FREQ_FINE      0x63   /* INTERNAL_FREQ_FINE, signed, ~0.15 % ODR per LSB */
#define LSM_FIFO_DATA_TAG  0x78   /* TAG, then X_L X_H Y_L Y_H Z_L Z_H at 0x79..0x7E */

#define LSM_WHO_AM_I_VAL   0x6C
#define LSM_TAG_XL_NC      0x02   /* TAG_SENSOR for accelerometer, non-compressed */

/* Configuration written by lsm_init() */
#define LSM_ODR_6667       0x0A   /* ODR / BDR code for 6667 Hz */
#define LSM_CTRL1_XL_VAL   ((LSM_ODR_6667 << 4) | (0x2 << 2))  /* 6667 Hz, +/-4 g, LPF2 off */
#define LSM_CTRL3_C_VAL    0x44   /* BDU=1, IF_INC=1 */
#define LSM_CTRL4_C_VAL    0x04   /* I2C_disable=1 (SPI only) */
#define LSM_FIFO_CTRL3_VAL LSM_ODR_6667          /* batch XL at 6667 Hz, gyro not batched */
#define LSM_FIFO_MODE_BYPASS     0x00
#define LSM_FIFO_MODE_CONTINUOUS 0x06
#define LSM_NOMINAL_ODR_HZ 6667.0f
#define LSM_G_PER_LSB_4G   0.000122f            /* 0.122 mg/LSB at +/-4 g */

typedef struct {
    int  (*read)(void *ctx, uint8_t reg, uint8_t *buf, uint16_t len);  /* 0 = ok */
    int  (*write)(void *ctx, uint8_t reg, uint8_t val);                /* 0 = ok */
    void (*delay_ms)(void *ctx, uint32_t ms);
    void *ctx;
} lsm_bus_t;

typedef struct {
    uint32_t xl_samples;      /* accelerometer samples delivered */
    uint32_t other_words;     /* FIFO words with an unexpected tag (should stay 0) */
    uint32_t tag_gaps;        /* TAG_CNT discontinuities: dropped words (mod-4 aliasing possible) */
    uint32_t overruns;        /* FIFO overrun flag seen: samples were lost */
    uint32_t bus_errors;
    uint8_t  last_cnt;
    uint8_t  have_last;
} lsm_stats_t;

enum { LSM_OK = 0, LSM_ERR_BUS = -1, LSM_ERR_WHOAMI = -2, LSM_ERR_RESET = -3 };

/* Reset and configure: 6667 Hz, +/-4 g, accel-only continuous FIFO. */
int lsm_init(const lsm_bus_t *bus);

/* Signed INTERNAL_FREQ_FINE; actual ODR ~= 6667 * (1 + 0.0015 * value). */
int lsm_freq_fine(const lsm_bus_t *bus, int8_t *value);

/* Read up to max_samples accelerometer samples (axis 0=x,1=y,2=z) in g from
 * the FIFO. Returns samples written (>= 0) or a negative error. */
int lsm_fifo_drain(const lsm_bus_t *bus, int axis, float *out, int max_samples,
                   lsm_stats_t *st);

#ifdef __cplusplus
}
#endif
#endif
