/** \file real_utilities.h
    \brief A simple template based interface to some common C float/double constants to ease writing of templated code.
*/

#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "core_defines.h"

#ifdef _USE_MATH_DEFINES
#include <math.h>
#else
#define _USE_MATH_DEFINES
#include <math.h>
#undef _USE_MATH_DEFINES
#endif

#include <float.h>

//
// Get scalar limits of operation
//

/* DPCT_ORIG template<class T> __inline__ __host__ __device__ T get_min();*/
template <class T> __inline__ T get_min();
/* DPCT_ORIG template<class T> __inline__ __host__ __device__ T get_max();*/
template <class T> __inline__ T get_max();
/* DPCT_ORIG template<class T> __inline__ __host__ __device__ T get_epsilon();*/
template <class T> __inline__ T get_epsilon();

//
// Math prototypes
//

/* DPCT_ORIG template<class REAL> __inline__ __host__ __device__ REAL get_pi();*/
template <class REAL> __inline__ REAL get_pi();

//
// Implementation
//

/* DPCT_ORIG template<> __inline__ __host__ __device__ float get_min<float>()*/
template <> __inline__ float get_min<float>()
{
  return FLT_MIN;
}

/* DPCT_ORIG template<> __inline__ __host__ __device__ double get_min<double>()*/
template <> __inline__ double get_min<double>()
{
  return DBL_MIN;
}

/* DPCT_ORIG template<> __inline__ __host__ __device__ float get_max<float>()*/
template <> __inline__ float get_max<float>()
{
  return FLT_MAX;
}

/* DPCT_ORIG template<> __inline__ __host__ __device__ double get_max<double>()*/
template <> __inline__ double get_max<double>()
{
  return DBL_MAX;
}

/* DPCT_ORIG template<> __inline__ __host__ __device__ float get_epsilon<float>()*/
template <> __inline__ float get_epsilon<float>()
{
  return FLT_EPSILON;
}

/* DPCT_ORIG template<> __inline__ __host__ __device__ double get_epsilon<double>()*/
template <> __inline__ double get_epsilon<double>()
{
  return DBL_EPSILON;
}

/* DPCT_ORIG template<> __inline__ __host__ __device__ float get_pi(){ return (float)M_PI; }*/
template <> __inline__ float get_pi() { return (float)M_PI; }
/* DPCT_ORIG template<> __inline__ __host__ __device__ double get_pi(){ return M_PI; }*/
template <> __inline__ double get_pi() { return M_PI; }
