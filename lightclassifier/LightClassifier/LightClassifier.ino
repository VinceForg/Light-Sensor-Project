// LightClassifier.ino -- Part 4: on-device inference on the ESP32.
//
// Put this file and model.h in a folder named LightClassifier/.
// Wiring (Figure 1): UV (GUVA-S12SD OUT) -> GPIO34, IR divider -> GPIO35.
//
// Every sample: read ADC -> /4095 -> EMA -> log1p(x*4095) -> scale with
// SCALE_BOUNDS -> grid cell -> one byte per class -> argmax -> reject if < tau.
// This MUST match the Python pipeline exactly (see the comments in model.h).
//
// Serial output (115200 baud), one CSV row per sample:
//   label,uv_raw,ir_raw,uv_filt,ir_filt,prediction,winning_value
// Type a label (e.g. "sun") in the Serial Monitor and press Enter to tag the
// following rows -- that is how you record the labelled held-out batch.
// Type "bench" to re-run the latency benchmark.

#include <Arduino.h>
#include <math.h>
#include "model.h"

const int UV_PIN = 34;            // ADC1_CH6 (sensor1)
const int IR_PIN = 35;            // ADC1_CH7 (sensor2)
const uint32_t SAMPLE_MS = 100;   // 10 Hz -- MUST match your data-logging rate

float uvFilt, irFilt;             // EMA state
String currentLabel = "none";

// ---- the online path (Algorithm 1) --------------------------------------

static inline int toCell(float v, float vmin, float vmax) {
  float s = (v - vmin) / (vmax - vmin);          // scale to [0, 1]
  int i = (int)floorf(s * GRID_N);
  if (i < 0) i = 0;
  if (i > GRID_N - 1) i = GRID_N - 1;            // clip to [0, G-1]
  return i;
}

// Returns class index, or -1 for UNKNOWN. winning gets the winning value v_c.
int classify(float uvF, float irF, float* winning) {
  float x = log1pf(uvF * 4095.0f);               // same transform as training
  float y = log1pf(irF * 4095.0f);
  int col = toCell(x, SCALE_BOUNDS[0], SCALE_BOUNDS[1]);
  int row = toCell(y, SCALE_BOUNDS[2], SCALE_BOUNDS[3]);

  int best = 0;
  uint8_t bestByte = 0;
  for (int c = 0; c < NUM_CLASSES; c++) {        // K byte reads
    uint8_t b = pgm_read_byte(&CLASS_TABLE[c][row][col]);
    if (b > bestByte) { bestByte = b; best = c; }
  }
  float v = bestByte / 255.0f;
  if (winning) *winning = v;
  return (v < REJECT_TAU) ? -1 : best;
}

// ---- latency benchmark ----------------------------------------------------

void benchmark() {
  const int N = 10000;
  volatile int sink = 0;                         // stops the compiler removing the loop
  float w;
  uint32_t t0 = micros();
  for (int i = 0; i < N; i++) {
    // vary the input so every call does real work
    sink += classify(uvFilt + i * 1e-7f, irFilt + i * 1e-7f, &w);
  }
  uint32_t dt = micros() - t0;
  Serial.printf("# latency: %d calls in %lu us -> %.3f us per query\n",
                N, (unsigned long)dt, (float)dt / N);
  Serial.printf("# model table: %d bytes (K*G^2 = %d*%d^2) + 16 bytes bounds\n",
                (int)sizeof(CLASS_TABLE), NUM_CLASSES, GRID_N);
  Serial.printf("# free heap: %u bytes\n", (unsigned)ESP.getFreeHeap());
}

// ---- Arduino entry points --------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(500);
  analogReadResolution(12);                      // 0..4095, same as data logging
  // analogSetAttenuation(ADC_11db);             // uncomment ONLY if your logging sketch set it

  uvFilt = analogRead(UV_PIN) / 4095.0f;         // start the EMA at the first reading
  irFilt = analogRead(IR_PIN) / 4095.0f;

  benchmark();
  Serial.println("label,uv_raw,ir_raw,uv_filt,ir_filt,prediction,winning_value");
}

void loop() {
  static uint32_t last = 0;

  if (Serial.available()) {                      // label / command input
    String in = Serial.readStringUntil('\n');
    in.trim();
    if (in == "bench") benchmark();
    else if (in.length()) currentLabel = in;
  }

  if (millis() - last < SAMPLE_MS) return;
  last = millis();

  int uvRaw = analogRead(UV_PIN);
  int irRaw = analogRead(IR_PIN);
  uvFilt = EMA_ALPHA * (uvRaw / 4095.0f) + (1.0f - EMA_ALPHA) * uvFilt;
  irFilt = EMA_ALPHA * (irRaw / 4095.0f) + (1.0f - EMA_ALPHA) * irFilt;

  float winning;
  int c = classify(uvFilt, irFilt, &winning);

  Serial.printf("%s,%d,%d,%.6f,%.6f,%s,%.3f\n",
                currentLabel.c_str(), uvRaw, irRaw, uvFilt, irFilt,
                c < 0 ? "UNKNOWN" : CLASS_NAMES[c], winning);
}
