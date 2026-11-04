/* Selects the model compiled into the ESP32 benchmark sketch.
 * Override with -DEM_MODEL_HEADER='"models/model_<fs>_<model>_<fold>.h"'.
 * Default: 6664 Hz MLP trained on loads 0-2, evaluated on held-out load 3. */
#ifndef EM_MODEL_CONFIG_H
#define EM_MODEL_CONFIG_H
#define EM_MODEL_HEADER "models/model_6664_mlp_load3.h"
#endif
