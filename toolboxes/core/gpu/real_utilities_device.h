#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>

//
// Math prototypes
//

template<class REAL> __inline__ void gad_sincos( REAL angle, REAL *a, REAL *b );
template<class REAL> __inline__ REAL gad_rsqrt( REAL val );


//
// Implementation
//

template<> __inline__ void gad_sincos<float>( float angle, float *a, float *b ){ sincosf(angle, a,b); }
template<> __inline__ void gad_sincos<double>( double angle, double *a, double *b ){ sincos(angle, a,b); }

template <> __inline__ float gad_rsqrt<float>(float val) { return sycl::rsqrt(val); }
template <> __inline__ double gad_rsqrt<double>(double val) { return sycl::rsqrt(val); }
