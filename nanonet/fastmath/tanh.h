#include "xmmintrin.h"

#define TANH_RANGE     4.5f
#define TANH_CLAMP     1.f
#define F21           21.f
#define F210         210.f
#define F1260       1260.f
#define F4725       4725.f
#define F10395     10395.f

__m128 sse_tanh(__m128 x){
  // tanh(x) = (21*x^5 + 1260*x^3 + 10395*x) /
  //           (x^6 + 210*x^4 + 4725*x^2 + 10395)
  __m128 y, s, d, i_d;
  size_t i;
  union {__m128 v; float f[4];} u1, u2;

  s = _mm_mul_ps(x, x);

  // numerator - (s*21 + 1260)*s + 10395)*x
  y = _mm_mul_ps(s, _mm_set1_ps(F21));
  y = _mm_add_ps(y, _mm_set1_ps(F1260));
  y = _mm_mul_ps(y, s);
  y = _mm_add_ps(y, _mm_set1_ps(F10395));
  y = _mm_mul_ps(y, x);

  // denominator - (s + 210)*s + 4725)*s + 10395)
  d = _mm_add_ps(s, _mm_set1_ps(F210));
  d = _mm_mul_ps(d, s);
  d = _mm_add_ps(d, _mm_set1_ps(F4725));
  d = _mm_mul_ps(d, s);
  d = _mm_add_ps(d, _mm_set1_ps(F10395));

  // reciprocal
  i_d = _mm_rcp_ps(d);
  i_d = _mm_sub_ps(_mm_add_ps(i_d, i_d), _mm_mul_ps(d, _mm_mul_ps(i_d, i_d)));

  u1.v = _mm_mul_ps(y, i_d);
  u2.v = x;
  for(i=0; i<4; ++i){
    if (u2.f[i] < -TANH_RANGE)
      u1.f[i] = -TANH_CLAMP;
    if (u2.f[i] > TANH_RANGE)
      u1.f[i] = TANH_CLAMP;
  }
  return u1.v;
}

