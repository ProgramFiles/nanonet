#include <Python.h>



#define MODULE_API_EXPORTS
#include "module.h"

#include <xmmintrin.h>
#include "tanh.h"

static PyMethodDef MathsMethods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initnanonetmaths(void) {
    (void) Py_InitModule("nanonetmaths", MathsMethods);
}


MODULE_API void fast_tanh(float* in, float* out, size_t size){
  union {__m128 v; float f[4];} u;
  size_t i;
  size_t j;

  for(i=0; i<size; i+=4){
    u.v = _mm_set_ps(in[i+3], in[i+2], in[i+1], in[i]);
    u.v = sse_tanh(u.v);
    for(j=0; j<4; ++j){
      out[i+j]=u.f[j];
    }
  }
}
