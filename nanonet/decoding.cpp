#include <Python.h>

#include <cassert>
#include <cstdlib>
#include <limits>
#include <vector>

#define MODULE_API_EXPORTS
#include "module.h"

typedef double ftype;
using namespace std;


static PyMethodDef DecodeMethods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initnanonetdecode(void) {
    (void) Py_InitModule("nanonetdecode", DecodeMethods);
}


extern "C" void viterbi_update(
  ftype* vit_last, ftype* vit_curr, int32_t* max_idx,
  const size_t num_bases, const size_t num_kmers,
  const ftype stay, const ftype step, const ftype skip, const ftype slip
){

  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    max_idx[kmer] = -1;
    vit_curr[kmer] = -std::numeric_limits<ftype>::infinity();
  }

  // Stay
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    if(vit_last[kmer]+stay>vit_curr[kmer]){
      vit_curr[kmer] = vit_last[kmer]+stay;
      max_idx[kmer] = kmer;
    }
  }
  // Step
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    const size_t idx = (kmer*num_bases)%num_kmers;
    for ( size_t i=0 ; i<num_bases ; i++){
      if(vit_last[kmer]+step>vit_curr[idx+i]){
        vit_curr[idx+i] = vit_last[kmer]+step;
        max_idx[idx+i] = kmer;
      }
    }
  }
  // Skip
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    const size_t idx = (kmer*num_bases*num_bases)%num_kmers;
    for ( size_t i=0 ; i<num_bases*num_bases ; i++){
      if(vit_last[kmer]+skip>vit_curr[idx+i]){
        vit_curr[idx+i] = vit_last[kmer]+skip;
        max_idx[idx+i] = kmer;
      }
    }
  }
  // Slip
  if (slip > -std::numeric_limits<ftype>::infinity()){
    ftype slip_max = -std::numeric_limits<ftype>::infinity();
    size_t slip_idx = 0;
    for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
      if(vit_last[kmer]+slip>slip_max){
        slip_max = vit_last[kmer]+slip;
        slip_idx = kmer;
      }
    }
    for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
      if(slip_max>vit_curr[kmer]){
        vit_curr[kmer] = slip_max;
        max_idx[kmer] = slip_idx;
      }
    }
  }
}


extern "C" double decode_path(ftype * logpost, const size_t num_events, const size_t num_bases, const size_t num_kmers){
  assert(NULL!=logpost);
  assert(num_events>0);
  assert(num_bases>0);
  assert(num_kmers>0);

  std::vector<int32_t> max_idx(num_kmers);
  std::vector<ftype> vit_last(num_kmers);
  std::vector<ftype> vit_curr(num_kmers);

  // Treat all movement types equally, disallow slip (allowing slip
  //   would simply give kmer with maximum posterioir)
  ftype stay = 0.0;
  ftype step = 0.0;
  ftype skip = 0.0;
  ftype slip = -std::numeric_limits<ftype>::infinity();

  // Initial values
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    vit_last[kmer] = logpost[kmer];
  }

  for ( size_t ev=1 ; ev<num_events ; ev++){
    const size_t idx1 = (ev-1)*num_kmers;
    const size_t idx2 = ev*num_kmers;

    viterbi_update(
      vit_last.data(), vit_curr.data(), max_idx.data(),
      num_bases, num_kmers,
      stay, step, skip, slip
    );

    // Emission
    for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
      vit_curr[kmer] += logpost[idx2+kmer];
    }

    // Traceback information
    for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
      logpost[idx1+kmer] = max_idx[kmer];
    }
    std::swap( vit_last, vit_curr );
  }

  // Decode states
  // Last state by Viterbi matrix
  const size_t idx = (num_events-1)*num_kmers;
  ftype max_val = -std::numeric_limits<ftype>::infinity();
  int max_kmer = -1;
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    if(vit_last[kmer]>max_val){
      max_val = vit_last[kmer];
      max_kmer = kmer;
    }
  }
  logpost[idx] = max_kmer;
  // Other states by traceback
  for ( size_t ev=(num_events-1) ; ev>0 ; ev--){
    const size_t idx = (ev-1)*num_kmers;
    logpost[idx] = logpost[idx+(int)logpost[idx+num_kmers]];
  }

  return max_val;
}


