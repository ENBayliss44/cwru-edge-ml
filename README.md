# cwru-edge-ml

An int8 bearing-fault classifier in plain C for the ESP32. From one second of vibration data it classifies the bearing as **normal, inner-race fault, outer-race fault or ball fault**. It is trained on the [CWRU bearing dataset](https://engineering.case.edu/bearingdatacenter) and targets the ESP32 Feather V2 + LSM6DSOX accelerometer from my EE284A health monitor, which used a fixed RMS threshold instead of a learned model.

## Key findings

- **Sample rate is the bottleneck.** The EE284A firmware reads the sensor over I2C at 1666 Hz. At that rate the best model reaches **77%** accuracy on motor loads it never trained on. At 6664 Hz (the LSM6DSOX maximum) it reaches **96%** and misses no faults. This firmware therefore [reads the sensor over SPI at 6667 Hz through its FIFO](#sensor-path-spi--fifo-at-6667-hz).
- **The fixed threshold doesn't generalize.** At 1666 Hz it detects 52% of faults and false-alarms on 17% of healthy windows.
- **int8 costs at most 0.2 points of accuracy**, and the C code reproduces Python exactly: bit-exact integer inference, with the same class predicted for 1,580 of 1,580 test windows.
- **Small footprint on the ESP32:** the classifier adds **14.5 KB of flash** and 96 KB of static RAM at 6664 Hz (12.4 KB and 24 KB at 1666 Hz), measured from the linker output.
- **Not yet measured:** ESP32 latency, and anything from the real sensor. All 6664 Hz accuracy results use resampled CWRU data.

## Results

The models are tested with **leave-one-load-out**: train on three motor loads (0–3 hp), test on the fourth, then pool the four folds. No recording appears in both training and test. Models are int8, and brackets show 95% Wilson intervals.

| Rate | Model | 4-class accuracy | Faults detected | False alarms |
|---|---|---|---|---|
| 1666 Hz (current) | RMS threshold | – | 188/360 (52%) | 6/35 (17%) |
| | Logistic regression | 73.7% [69, 78] | 335/360 (93%) | 4/35 (11%) |
| | MLP 23→16→4 | 77.0% [73, 81] | 336/360 (93%) | 5/35 (14%) |
| 6664 Hz (sensor max) | RMS threshold | – | 360/360 (100%) | 0/35 (0%) |
| | Logistic regression | 92.4% [89, 95] | 360/360 (100%) | 2/35 (6%) |
| | MLP 23→16→4 | **95.9%** [94, 97] | 360/360 (100%) | 2/35 (6%) [2, 19] |

At 6664 Hz the threshold already detects every fault, so the model's contribution is identifying *which* fault it is.

**Confusion matrix, MLP**: each cell shows 1666 Hz / 6664 Hz counts. Rows are the true class, columns the predicted class.

| | normal | inner | outer | ball |
|---|---|---|---|---|
| **normal** | 30 / 33 | 4 / 0 | 0 / 2 | 1 / 0 |
| **inner** | 0 / 0 | 106 / 117 | 9 / 0 | 5 / 3 |
| **outer** | 0 / 0 | 14 / 0 | 106 / 120 | 0 / 0 |
| **ball** | **24** / 0 | 33 / 3 | 1 / 8 | 62 / 109 |

At 1666 Hz, 24 ball-fault windows are classified as normal. These are missed faults, the dangerous failure mode. At 6664 Hz no fault window is classified as normal.

**Sample-rate ablation**: same pipeline and hyperparameters at every rate, float models.

| Rate | Threshold (detected / false alarms) | MLP accuracy | MLP accuracy, unseen load **and** fault size |
|---|---|---|---|
| 1666 Hz | 52% / 17% | 77.0% | 44.7% |
| 3332 Hz | 66% / 0% | 81.3% | 46.0% |
| 6664 Hz | 100% / 0% | 95.7% | 50.8% |
| 12000 Hz | 100% / 14% | 100% | 66.9% |

CWRU fault impacts excite structural resonances at roughly 2–4 kHz, which 1666 Hz sampling (833 Hz Nyquist) can't capture. For comparison, a random window split, which is common in CWRU papers, gives 98.4% at 1666 Hz, because correlated windows from the same recording end up in both training and test.

## Sensor path: SPI + FIFO at 6667 Hz

The EE284A firmware calls `lsm.getEvent()` once per sample over I2C, and that per-call overhead caps the rate at 1666 Hz. This firmware instead reads the LSM6DSOX over **SPI** and drains its **FIFO** in bursts. At 6667 Hz the FIFO data is about 0.4 Mbit/s, under 5% of an 8 MHz SPI bus; I2C at 400 kHz can't carry it.

- **Driver** (`src/sensor/lsm6dsox_fifo.c`): a register-level C driver with no Adafruit dependency. It configures the accelerometer at 6667 Hz and ±4 g, powers the gyroscope down, sends accelerometer data to a continuous FIFO, and disables I2C. On every read it checks each FIFO word's sensor tag, detects dropped samples from the tag counter, and reports overruns.
- **Real-time structure:** a FreeRTOS task on core 0 drains the FIFO every 2 ms into two ping-pong window buffers, while core 1 classifies whichever 6664-sample window was completed last. If the classifier ever falls a full window behind, the window is dropped and counted. Samples are never overwritten while being classified.
- **Wiring:** the sensor has its own SPI bus, because the EE284A thermocouple uses the Feather's default SPI pins. Connect SCK→GPIO 14, SDI→32, SDO→33 and CS→27 (details in `src/sensor/sensor_config.h`), and disconnect the sensor from the STEMMA QT/I2C chain.

**Verified so far:**
- The driver passes host tests against a simulated LSM6DSOX (`firmware/host/test_sensor.c`). The tests check the register sequence, sample rate and units on all three axes, injected tag gaps and FIFO overruns, the ping-pong handoff, and a full window run through the C feature extractor. Mutation checks confirm the tests fail when the output rate, gap detection or overrun detection is broken.
- All firmware variants compile for the Feather V2 with no warnings.

**Not yet verified on hardware:** the simulation is built from my reading of the datasheet, so it can't catch a misread register. Before making device claims, run `tools/sensor_check.py <port> selftest`. It passes only if the measured sample rate is within 0.5% of the rate implied by the sensor's own trim register, with zero overruns, gaps, unexpected FIFO words and bus errors. Two things also remain to be checked on hardware: the sensor's usable bandwidth at this rate, and its noise floor compared with CWRU's piezoelectric accelerometers.

## Method

- **Data.** 40 CWRU 12 kHz drive-end recordings: 4 healthy, plus inner-race, ball and outer-race faults at 0.007, 0.014 and 0.021 in, each at loads of 0–3 hp (see `data/manifest.csv`).
- **Sample-rate correction.** The healthy files (97–100) are **48 kHz, not 12 kHz**. At 48 kHz their speed-locked rig peaks match the 12 kHz fault files: 155.27 Hz in both `100.mat` and `133.mat`, both at 1725 rpm. Healthy files are decimated to 12 kHz, then resampled along the same path as the fault files.
- **Features (23 per 1 s window).** 7 time-domain features (log RMS, log peak-to-peak, crest factor, kurtosis, skewness, impulse factor, shape factor) and 16 log band powers from a Hann-windowed FFT covering 0 to fs/2.
- **Models.** Logistic regression and an MLP with 16 ReLU units (452 parameters), with classes balanced and hyperparameters fixed in advance.
- **int8.** Weights are per-tensor int8 and biases int32. The hidden layer is requantized with a fixed-point (Q31) multiplier, and the prediction is the argmax of the int32 logits. `src/quantize.py` and the C code implement the same integer arithmetic.
- **C library** (`firmware/edge_ml_bench/src/edge_ml/`): C99, no heap, depends only on `math.h`, with a radix-2 FFT. It was verified against Python on a Mac across 16 models (2 rates × 2 models × 4 folds), compiled with `clang -Wall -Wextra -Werror`. Only 2 of 36,340 int8 inputs differed, each by 1 LSB, and no prediction changed.
- **ESP32 firmware** (`edge_ml_bench.ino`) has two inputs. It can take test windows streamed over serial (`tools/stream_bench.py` compares the outputs with Python and records `esp_timer` latency; not yet run), or it can classify live from the sensor. Flash and RAM costs are measured as the difference between a full build and an ML-stubbed build (`tools/build_firmware.sh`).

## Limitations

- **One rig, artificial faults.** CWRU faults were cut with electrical discharge machining rather than grown naturally, and healthy and fault data were recorded in different sessions.
- **Few healthy windows.** There are only 35 healthy test windows, so false-alarm rates are uncertain: 2/35, 95% CI 2–19%.
- **Unseen fault sizes don't generalize**: 45–51% accuracy at 1666–6664 Hz.
- **Untested on the real sensor.** The model has never seen LSM6DSOX data, so it needs validation with the actual sensor and mounting. The 6664 Hz figures come from resampled data until the SPI path passes its hardware self-test.
- **No "unknown" output.** A safety-critical deployment should add a confidence margin and keep the threshold as an independent check.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn matplotlib pyserial
cd src
../.venv/bin/python download.py               # CWRU data (~145 MB)
../.venv/bin/python experiments.py            # split protocols
../.venv/bin/python experiments.py ablation   # sample-rate ablation
../.venv/bin/python quantize.py               # float vs int8
../.venv/bin/python export_verify.py          # C headers + host C verification
../.venv/bin/python report.py                 # numbers in this README
```

```bash
cd ..
./tools/run_host_tests.sh                     # sensor driver vs. simulated LSM6DSOX
./tools/build_firmware.sh                     # ESP32 builds + flash/RAM deltas (needs arduino-cli, esp32 core)
```

On hardware, flash `firmware/edge_ml_bench/`, then run:

```bash
.venv/bin/python tools/sensor_check.py <port> selftest     # sample rate, overruns, gaps
.venv/bin/python tools/sensor_check.py <port> live 60      # live classification log
.venv/bin/python tools/stream_bench.py <port> 6664_mlp_load3   # latency + Python cross-check
```

Results are written to `results/`.

Data: Case Western Reserve University Bearing Data Center.
