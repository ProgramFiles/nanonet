#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <err.h>
#include <stdint.h>


#define MODULE_API_EXPORTS
#include "module.h"

#include <Python.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

/**
 * For setuptools on windows. Since we aren't going the
 * python.h route we will still have to load this library via
 * ctypes, that's fine because we use ctypes for passing numpy
 * arrrays anyway. Note, the function name after "init" is tied
 * to the Extension name in setup.py
**/
void initclib_viterbi() {}



//#include <common.h>
#include <filters.h>





/**
 *   Compute cumulative sum and sum of squares for a vector of data
 *   data      double[d_length]   Data to be summed over (in)
 *   sum       double[d_length]   Vector to store sum (out)
 *   sumsq     double[d_length]   Vector to store sum of squares (out)
 *   d_length                     Length of data vector
 **/
MODULE_API void compute_sum_sumsq(const double * restrict data, double* restrict sum, double* restrict sumsq, size_t d_length) {
  // Basic contracts
  assert(NULL!=data);
  assert(NULL!=sum);
  assert(NULL!=sumsq);
  assert(d_length>0);

  // Code
  sum[0] = data[0];
  sumsq[0] = data[0]*data[0];
  for (size_t i = 1; i < d_length; ++i) {
    sum[i] = sum[i - 1] + data[i];
    sumsq[i] = sumsq[i - 1] + data[i]*data[i];
  }
}

/**
 *    Compute moving average over window, output centred on current coordinate
 *    sum      double[d_length] Input data, cumulative sum (in)
 *    out      double[d_length] Ouput data (out)
 *    d_length Length of data vector
 *    w_length Length of window to compute mave over.  Made odd if not.
 **/
MODULE_API void compute_mave(const double* restrict sum, double* restrict mave, size_t d_length, size_t w_length) {
  // Simple contracts
  assert(NULL!=sum);
  assert(NULL!=mave);
  assert(d_length>0);
  assert(w_length>0);
  // make window length odd
  if(w_length % 2 == 0){
      w_length -= 1;
  }


  // quick return
  if (d_length < w_length || w_length < 2) {
    mave[0] = sum[0];
    for(size_t i = 1; i < d_length; ++i)
      mave[i] = sum[i] - sum[i-1];
    return;
  }

  size_t h_length = w_length/2;
  // fudge boundaries
  for(size_t i = 0; i < h_length; ++i) {
    mave[i] = (sum[i+h_length]) / (i+1+h_length);

    size_t ip = d_length - 1 - i;
    mave[ip] = (sum[d_length - 1] - sum[ip-h_length-1]) / (i+1+h_length);
  }
  // most of the data
  for(size_t i = h_length; i < d_length - h_length ; ++i) {
    mave[i] = (sum[i+h_length] - sum[i-h_length-1]) / (w_length);
  }
  return;
}


/**
 *   Compute windowed t-statistic from summary information
 *   sum       double[d_length]  Cumulative sums of data (in)
 *   sumsq     double[d_length]  Cumulative sum of squares of data (in)
 *   tstat     double[d_length]  T-statistic (out)
 *   d_length                    Length of data vector
 *   w_length                    Window length to calculate t-statistic over
 **/
MODULE_API void compute_tstat(const double* restrict sum, const double* restrict sumsq, double* restrict tstat, size_t d_length, size_t w_length, bool pooled) {
  // Simple contracts
  assert(NULL!=sum);
  assert(NULL!=sumsq);
  assert(NULL!=tstat);
  double eta = 1e-100;

  // Quick return:
  //   t-test not defined for number of points less than 2
  //   need at least as many points as twice the window length
  if (d_length < 2*w_length || w_length < 2) {
    for(size_t i = 0; i < d_length; ++i){
      tstat[i] = 0.0;
    }
    return;
  }

  // fudge boundaries
  for (size_t i = 0; i < w_length; ++i) {
    tstat[i] = 0;
    tstat[d_length - i - 1] = 0;
  }

  // get to work on the rest
  for (size_t i = w_length; i <= d_length - w_length; ++i) {
    double sum1 = sum[i - 1];
    double sumsq1 = sumsq[i - 1];
    if (i > w_length) {
      sum1 -= sum[i - w_length - 1];
      sumsq1 -= sumsq[i - w_length - 1];
    }
    double sum2 = sum[i + w_length - 1] - sum[i - 1];
    double sumsq2 = sumsq[i + w_length - 1] - sumsq[i - 1];
    double mean1 = sum1 / w_length;
    double mean2 = sum2 / w_length;
    double var1 = sumsq1 / w_length - mean1*mean1;
    double var2 = sumsq2 / w_length - mean2*mean2;
    if(pooled){
      var1 = ( var1 + var2 ) / 2.0;
      var2 = var1;
    }
    // Prevent problem due to very small variances
    var1 = fmax(var1, eta);
    var2 = fmax(var2, eta);
    const double totvar = var1 / w_length + var2 / w_length;

    //t-stat
    //  Formula is a simplified version of Student's t-statistic for the
    //  special case where there are two samples of equal size with
    //  differing variance
    const double delta = mean2 - mean1;
    tstat[i] = fabs(delta / sqrt(totvar));
  }
}


/**
 *   Compute windowed deltamean value from summary information
 *   sum       double[d_length]  Cumulative sums of data (in)
 *   sumsq     double[d_length]  Cumulative sum of squares of data (in)
 *   deltamean     double[d_length]  deltamean (out)
 *   d_length                    Length of data vector
 *   w_length                    Window length to calculate t-statistic over
 **/

MODULE_API void compute_deltamean(const double* restrict sum, const double* restrict sumsq, double* restrict deltamean, size_t d_length, size_t w_length) {

  // Set boundaries to 0.
  for (size_t i = 0; i < w_length; ++i) {
    deltamean[i] = 0;
    deltamean[d_length - i - 1] = 0;
  }

  // compute deltamean for non-boundary data
  for (size_t i = w_length; i <= d_length - w_length; ++i) {
    double sum1 = sum[i - 1];
    if (i > w_length) {
      sum1 -= sum[i - w_length - 1];
    }
    double sum2 = sum[i + w_length - 1] - sum[i - 1];
    double mean1 = sum1 / w_length;
    double mean2 = sum2 / w_length;

    const double delta = mean2 - mean1;
    // assume variance of 1.0 - approximately correct and avoids extra division
    deltamean[i] = fabs(delta);
  }
}


MODULE_API void short_long_peak_detector(const restrict DetectorPtr short_detector, const restrict DetectorPtr long_detector, const double peak_height, size_t* restrict peaks){
  assert(short_detector->signal_length == long_detector->signal_length);
  assert(NULL!=peaks);

  DetectorPtr detectors[2] = {short_detector, long_detector};
  size_t peak_count = 0;


  for(size_t i=0; i<short_detector->signal_length; i++){
    for(size_t k=0; k<2; k++){
      DetectorPtr detector = detectors[k];
      //Carry on if we've been masked out
      if (detector->masked_to >= i){
        continue;
      }

      double current_value = detector->signal[i];

      if (detector->peak_pos == detector->DEF_PEAK_POS){
        //CASE 1: We've not yet recorded a maximum
        if (current_value < detector->peak_value){
          //Either record a deeper minimum...
          detector->peak_value = current_value;
        }
        else if (current_value - detector->peak_value > peak_height){
          // ...or we've seen a qualifying maximum
          detector->peak_value = current_value;
          detector->peak_pos = i;
          //otherwise, wait to rise high enough to be considered a peak
        }
      }
      else {
        //CASE 2: In an existing peak, waiting to see if it is good
        if (current_value > detector->peak_value){
          //Update the peak
          detector->peak_value = current_value;
          detector->peak_pos = i;
        }

        //Dominate other tstat signals if we're going to fire at some point
        if (detector == short_detector){
          if (detector->peak_value > detector->threshold){
            long_detector->masked_to = detector->peak_pos + detector->window_length;
            long_detector->peak_pos = long_detector->DEF_PEAK_POS;
            long_detector->peak_value = long_detector->DEF_PEAK_VAL;
            long_detector->valid_peak = false;
          }
        }

        //Have we convinced ourselves we've seen a peak
        if (detector->peak_value - current_value > peak_height && detector->peak_value > detector->threshold){
          detector->valid_peak = true;
        }

        //Finally, check the distance if this is a good peak
        if (detector->valid_peak && (i - detector->peak_pos) > detector->window_length / 2){
          //Emit the boundary and reset
          peaks[peak_count] = detector->peak_pos;
          peak_count++;
          detector->peak_pos = detector->DEF_PEAK_POS;
          detector->peak_value = current_value;
          detector->valid_peak = false;
        }
      }
    }
  }
}

