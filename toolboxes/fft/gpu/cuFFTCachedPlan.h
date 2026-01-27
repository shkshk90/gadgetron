#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuFFTPlan.h"
#include <map>

namespace Gadgetron {
template <class ComplexType> class cuFFTCachedPlan {
  public:
    cuFFTCachedPlan() = default;

    void fft1(cuNDArray<ComplexType>& in_out, bool scale=true);
    void fft2(cuNDArray<ComplexType>& in_out, bool scale=true);
    void fft3(cuNDArray<ComplexType>& in_out, bool scale=true);

    void ifft1(cuNDArray<ComplexType>& in_out, bool scale=true);
    void ifft2(cuNDArray<ComplexType>& in_out, bool scale=true);
    void ifft3(cuNDArray<ComplexType>& in_out, bool scale=true);

    void fft1c(cuNDArray<ComplexType>& in_out, bool scale=true);
    void fft2c(cuNDArray<ComplexType>& in_out, bool scale=true);
    void fft3c(cuNDArray<ComplexType>& in_out, bool scale=true);

    void ifft1c(cuNDArray<ComplexType>& in_out, bool scale=true);
    void ifft2c(cuNDArray<ComplexType>& in_out, bool scale=true);
    void ifft3c(cuNDArray<ComplexType>& in_out, bool scale=true);

    void fft(cuNDArray<ComplexType>& in_out, unsigned int rank, bool scale=true);
    void ifft(cuNDArray<ComplexType>& in_out, unsigned int rank, bool scale=true);

    void fftc(cuNDArray<ComplexType>& in_out, unsigned int rank, bool scale=true);
    void ifftc(cuNDArray<ComplexType>& in_out, unsigned int rank, bool scale=true);

  private:
    std::map<std::vector<size_t>, cuFFTPlan<ComplexType, void >> plans;
};

} // namespace Gadgetron

#include "cuNDArray_math.h"

namespace {
namespace cufft_detail{
    auto compact_dims(int rank, const std::vector<size_t>& dimensions){
        assert(rank <= dimensions.size());
        auto dims = std::vector<size_t>(dimensions.begin(),dimensions.begin()+rank);
        auto batches = std::reduce(dimensions.begin()+rank,dimensions.end(),1,std::multiplies());
        dims.push_back(batches);
        return dims;
    }

    auto fetch_plan = [](auto& cache, const auto& compacted_dims ) -> auto& {
        return cache.try_emplace(compacted_dims,compacted_dims.size()-1,compacted_dims).first->second;
    };
}
}

template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::fft1(cuNDArray<ComplexType>& in_out, bool scale) {
    this->fft(in_out, 1, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::fft2(cuNDArray<ComplexType>& in_out, bool scale) {
    this->fft(in_out, 2, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::fft3(cuNDArray<ComplexType>& in_out, bool scale) {
    this->fft(in_out, 3, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::ifft1(cuNDArray<ComplexType>& in_out, bool scale) {
    this->ifft(in_out, 1, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::ifft2(cuNDArray<ComplexType>& in_out, bool scale) {
    this->ifft(in_out, 2, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::ifft3(cuNDArray<ComplexType>& in_out, bool scale) {
    this->ifft(in_out, 3, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::fft1c(cuNDArray<ComplexType>& in_out, bool scale) {
    this->fftc(in_out, 1, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::fft2c(cuNDArray<ComplexType>& in_out, bool scale) {
    this->fftc(in_out, 2, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::fft3c(cuNDArray<ComplexType>& in_out, bool scale) {
    this->fftc(in_out, 3, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::ifft1c(cuNDArray<ComplexType>& in_out, bool scale) {
    this->ifftc(in_out, 1, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::ifft2c(cuNDArray<ComplexType>& in_out, bool scale) {
    this->ifftc(in_out, 2, scale);
}
template <class ComplexType> void Gadgetron::cuFFTCachedPlan<ComplexType>::ifft3c(cuNDArray<ComplexType>& in_out, bool scale) {
    this->ifftc(in_out, 3, scale);
}

template <class ComplexType>
void Gadgetron::cuFFTCachedPlan<ComplexType>::fft(Gadgetron::cuNDArray<ComplexType>& in_out, unsigned int rank, bool scale) {
    cufft_detail::fetch_plan(plans,cufft_detail::compact_dims(rank, in_out.dimensions())).fft(in_out, scale);
}
template <class ComplexType>
void Gadgetron::cuFFTCachedPlan<ComplexType>::ifft(Gadgetron::cuNDArray<ComplexType>& in_out, unsigned int rank, bool scale) {

    cufft_detail::fetch_plan(plans,cufft_detail::compact_dims(rank, in_out.dimensions())).ifft(in_out, scale);
}
template <class ComplexType>
void Gadgetron::cuFFTCachedPlan<ComplexType>::fftc(Gadgetron::cuNDArray<ComplexType>& in_out, unsigned int rank, bool scale) {
    cufft_detail::fetch_plan(plans,cufft_detail::compact_dims(rank, in_out.dimensions())).fftc(in_out, scale);
}
template <class ComplexType>
void Gadgetron::cuFFTCachedPlan<ComplexType>::ifftc(Gadgetron::cuNDArray<ComplexType>& in_out, unsigned int rank, bool scale) {
    cufft_detail::fetch_plan(plans,cufft_detail::compact_dims(rank, in_out.dimensions())).ifftc(in_out, scale);
}
