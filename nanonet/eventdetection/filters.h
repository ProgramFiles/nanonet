#ifndef FILTERS_H
#define FILTERS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
 int DEF_PEAK_POS;
 double DEF_PEAK_VAL;
 double * signal;
 size_t signal_length;
 double threshold;
 size_t window_length;
 size_t masked_to;
 int peak_pos;
 double peak_value;
 _Bool valid_peak;
} Detector;
typedef Detector * DetectorPtr;

void short_long_peak_detector(const DetectorPtr short_detect, const DetectorPtr long_detector, const double peak_height, size_t* restrict peaks);


#endif /* FILTERS_H */
