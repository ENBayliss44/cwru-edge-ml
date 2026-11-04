/* LSM6DSOX SPI wiring and sampling settings (Adafruit ESP32 Feather V2).
 *
 * The sensor gets its own SPI bus (HSPI) because the EE284A MAX6675
 * thermocouple driver bit-bangs GPIO 5 and 21 (the Feather's default SPI SCK
 * and MISO). GPIO 12 and 15 are boot-strapping pins and are avoided.
 *
 *   LSM6DSOX breakout   Feather V2
 *   SCL  (SPI SCK)  ->  GPIO 14
 *   SDA  (SPI SDI)  ->  GPIO 32   (MOSI)
 *   DO   (SPI SDO)  ->  GPIO 33   (MISO)
 *   CS              ->  GPIO 27
 *   VIN / GND       ->  3V / GND
 * Disconnect the sensor from the STEMMA QT (I2C) chain when using SPI.
 */
#ifndef SENSOR_CONFIG_H
#define SENSOR_CONFIG_H

#define EM_SPI_SCK   14
#define EM_SPI_MOSI  32
#define EM_SPI_MISO  33
#define EM_SPI_CS    27
#ifndef EM_SPI_HZ
#define EM_SPI_HZ    8000000   /* LSM6DSOX max is 10 MHz; 8 MHz leaves margin for jumper wires */
#endif

#ifndef EM_SENSOR_AXIS
#define EM_SENSOR_AXIS 2       /* 0=x 1=y 2=z: choose the axis radial to the bearing */
#endif
#define EM_POLL_MS        2    /* FIFO drain period: ~13 samples per poll at 6667 Hz */
#define EM_SETTLE_SAMPLES 667  /* discard first ~100 ms after start-up */
#define EM_SELFTEST_S     10

#endif
