#pragma once
#include <Arduino.h>
#define GRID_N 1
#define NUM_CLASSES 4
#define REJECT_TAU 0.12f
#define EMA_ALPHA 0.2f
const uint8_t CLASS_TABLE[NUM_CLASSES][GRID_N][GRID_N] PROGMEM = {{{0}}, {{0}}, {{0}}, {{0}}};
const float SCALE_BOUNDS[4] = {0, 1, 0, 1};
const char* CLASS_NAMES[NUM_CLASSES] = {"shade", "sun", "indoor", "shade"};