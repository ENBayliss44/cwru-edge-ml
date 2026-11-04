# cwru-edge-ml

An int8 bearing-fault classifier in plain C for the ESP32. From one second of vibration data it classifies the bearing as **normal, inner-race fault, outer-race fault or ball fault**. It is trained on the [CWRU bearing dataset](https://engineering.case.edu/bearingdatacenter) and targets the ESP32 Feather V2 + LSM6DSOX accelerometer from my EE284A health monitor, which used a fixed RMS threshold instead of a learned model.

## Key findings

- **Sample rate is the bottleneck.** The EE284A firmware reads the sensor over I2C at 1666 Hz. At that rate the best model reaches **77%** accuracy on motor loads it never trained on. At 6664 Hz (the LSM6DSOX maximum) it reaches **96%** and misses no faults. Next step: [read the sensor over SPI at its maximum rate](#next-step-spi--fifo-at-the-sensor-maximum).
- **The fixed threshold doesn't generalize.** At 1666 Hz it detects 52% of faults and false-alarms on 17% of healthy windows.
- **int8 costs at most 0.2 points of accuracy**, and the C code reproduces Python exactly: bit-exact integer inference, with the same class predicted for 1,580 of 1,580 test windows.
- **Not yet measured:** ESP32 latency, flash and RAM, and anything from the real sensor. All 6664 Hz results use resampled CWRU data.

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

## Next step: SPI + FIFO at the sensor maximum

The current firmware calls `lsm.getEvent()` once per sample over I2C. Each call is a separate multi-byte I2C read that also reads the gyroscope and temperature, and that per-call overhead caps the rate at 1666 Hz. The plan is to move the LSM6DSOX to **SPI** and drain its FIFO in bursts:

1. **Wire the sensor over SPI** using the Feather's hardware SPI pins plus a GPIO for chip select, and initialise it with `begin_SPI()`, setting the clock explicitly up to the sensor's 10 MHz limit. At 6667 Hz, 3-axis FIFO data is about 47 kB/s (~0.4 Mbit/s), under 5% of a 10 MHz bus. I2C at 400 kHz can't carry that.
2. **Set the accelerometer to 6667 Hz and send it to the FIFO**, with the gyroscope off. Read the FIFO in bursts at a watermark (interrupt on INT1). This gives evenly spaced samples, even while the CPU is busy running inference.
3. **Assemble 6664-sample windows** and run the 6664 Hz model, which is already exported. The C pipeline only changes `EM_WINDOW` and `EM_NFFT`.

**To verify on hardware before making device claims:** the achieved sample rate with no FIFO overruns, the sensor's usable bandwidth at this rate, its noise floor compared with CWRU's piezoelectric accelerometers, and latency and RAM with the larger buffers (96 KB static for the 8192-point FFT, plus a 27 KB window buffer).

## Method

- **Data.** 40 CWRU 12 kHz drive-end recordings: 4 healthy, plus inner-race, ball and outer-race faults at 0.007, 0.014 and 0.021 in, each at loads of 0–3 hp (see `data/manifest.csv`).
- **Sample-rate correction.** The healthy files (97–100) are **48 kHz, not 12 kHz**. At 48 kHz their speed-locked rig peaks match the 12 kHz fault files: 155.27 Hz in both `100.mat` and `133.mat`, both at 1725 rpm. Healthy files are decimated to 12 kHz, then resampled along the same path as the fault files.
- **Features (23 per 1 s window).** 7 time-domain features (log RMS, log peak-to-peak, crest factor, kurtosis, skewness, impulse factor, shape factor) and 16 log band powers from a Hann-windowed FFT covering 0 to fs/2.
- **Models.** Logistic regression and an MLP with 16 ReLU units (452 parameters), with classes balanced and hyperparameters fixed in advance.
- **int8.** Weights are per-tensor int8 and biases int32. The hidden layer is requantized with a fixed-point (Q31) multiplier, and the prediction is the argmax of the int32 logits. `src/quantize.py` and the C code implement the same integer arithmetic.
- **C library** (`firmware/edge_ml_bench/src/edge_ml/`): C99, no heap, depends only on `math.h`, with a radix-2 FFT. It was verified against Python on a Mac across 16 models (2 rates × 2 models × 4 folds), compiled with `clang -Wall -Wextra -Werror`. Only 2 of 36,340 int8 inputs differed, each by 1 LSB, and no prediction changed.
- **ESP32 benchmark** (pending): `edge_ml_bench.ino` receives test windows over serial and reports the class and `esp_timer` latency. `tools/stream_bench.py` compares the results with Python. Flash and RAM are measured as the difference between a full build and an ML-stubbed build.

## Limitations

- **One rig, artificial faults.** CWRU faults were cut with electrical discharge machining rather than grown naturally, and healthy and fault data were recorded in different sessions.
- **Few healthy windows.** There are only 35 healthy test windows, so false-alarm rates are uncertain: 2/35, 95% CI 2–19%.
- **Unseen fault sizes don't generalize**: 45–51% accuracy at 1666–6664 Hz.
- **Untested on the real sensor.** The model has never seen LSM6DSOX data, so it needs validation with the actual sensor and mounting. The 6664 Hz figures are simulated until the SPI path is verified.
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

Results are written to `results/`. For the device benchmark, flash `firmware/edge_ml_bench/` and run `tools/stream_bench.py <port> 6664_mlp_load3`.

Data: Case Western Reserve University Bearing Data Center.
