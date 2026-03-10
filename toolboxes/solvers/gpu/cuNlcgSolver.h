#pragma once

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "nlcgSolver.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_blas.h"
#include "real_utilities.h"
#include "vector_td_utilities.h"
#include "gpusolvers_export.h"
#include <dpct/dpl_utils.hpp>

#include <functional>

#include "cuSolverUtils.h"

namespace Gadgetron{
  
  template <class T> class cuNlcgSolver : public nlcgSolver<cuNDArray<T> >
  {
  public:
    cuNlcgSolver() : nlcgSolver<cuNDArray<T> >() {}
    virtual ~cuNlcgSolver() {}
  };
}
