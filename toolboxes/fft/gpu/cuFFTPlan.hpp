#pragma once
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDFFT.h"
#include <numeric>

namespace Gadgetron::FFT_internal {
    namespace {
        template <class T> constexpr dpct::fft::fft_type transform_type() {
            if constexpr (std::is_same_v<float, Gadgetron::realType_t<T>>)
                return dpct::fft::fft_type::complex_float_to_complex_float;
            if constexpr (std::is_same_v<double, Gadgetron::realType_t<T>>)
                return dpct::fft::fft_type::complex_double_to_complex_double;
        }


        bool compatible_dimensions(int rank, const std::vector<size_t> &dim1, const std::vector<size_t> &dim2) {

            auto count_batches = [&](const auto &dim) {
                return std::accumulate(dim.begin() + rank, dim.end(), 1, std::multiplies());
            };

            return std::equal(dim1.begin(), dim1.begin() + rank, dim2.begin()) &&
                   count_batches(dim1) == count_batches(dim2);

        };

        std::string CUFFT_error_string(int error) {
            switch (error) {
                case 0:
                    return "CUFFT_SUCCESS";
                case 1:
                    return "CUFFT_INVALID_PLAN";
                case 2:
                    return "CUFFT_ALLOC_FAILED";
                case 3:
                    return "CUFFT_INVALID_TYPE";
                case 4:
                    return "CUFFT_INVALID_VALUE";
                case 5:
                    return "CUFFT_INTERNAL_ERROR";
                case 6:
                    return "CUFFT_EXEC_FAILED";
                case 7:
                    return "CUFFT_SETUP_FAILED";
                case 8:
                    return "CUFFT_INVALID_SIZE";
                case 9:
                    return "CUFFT_UNALIGNED_DATA";
                case 10:
                    return "CUFFT_INCOMPLETE_PARAMETER_LIST";
                case 11:
                    return "CUFFT INVALID_DEVICE";
                case 12:
                    return "CUFFT_PARSE_ERROR";
                case 13:
                    return "CUFFT_NO_WORKSPACE";
                case 14:
                    return "CUFFT_NOT_IMPLEMENTED";
                case 15:
                    return "CUFFT_LICENSE_ERROR";
                case 16:
                    return "CUFFT_NOT_SUPPORTED";
            }

            return "<unknown>";
        }

        template <class ComplexType>
        int executePlan(dpct::fft::fft_engine_ptr handle,
                        const ComplexType *idata, ComplexType *odata,
                        int direction) {
            if constexpr (std::is_same_v<float, Gadgetron::realType_t<ComplexType>>) {
                return DPCT_CHECK_ERROR(
                    (handle->compute<sycl::float2, sycl::float2>(
                        (sycl::float2 *)idata, (sycl::float2 *)odata,
                        direction == 1 ? dpct::fft::fft_direction::backward
                                       : dpct::fft::fft_direction::forward)));
            }
            if constexpr (std::is_same_v<double, Gadgetron::realType_t<ComplexType>>) {
                return DPCT_CHECK_ERROR(
                    (handle->compute<sycl::double2, sycl::double2>(
                        (sycl::double2 *)idata, (sycl::double2 *)odata,
                        direction == 1 ? dpct::fft::fft_direction::backward
                                       : dpct::fft::fft_direction::forward)));
            }
            return 16;
        }

        template<class ComplexType>
        void timeswitch(cuNDArray<ComplexType> &in_out, int rank) {

            switch (rank) {
                case 1:
                    return Gadgetron::timeswitch1D(&in_out);
                case 2:
                    return Gadgetron::timeswitch2D(&in_out);
                default:
                    Gadgetron::timeswitch3D(&in_out);
            }
            for (int i = 3; i < rank; i++)
                Gadgetron::timeswitch(&in_out, i);
        }
    } // namespace
}

template<class ComplexType, class ENABLER>
Gadgetron::cuFFTPlan<ComplexType, ENABLER>::cuFFTPlan(int rank, const std::vector<size_t> &dimensions) : rank(rank),
                                                                                                         dimensions(
                                                                                                                 dimensions) {

    if (rank > dimensions.size())
        throw std::invalid_argument("Rank must be equal or smaller than the number of dimensions given ");

    auto int_dimensions = std::vector<int>(dimensions.begin(), dimensions.begin() + rank);
    std::reverse(int_dimensions.begin(), int_dimensions.end());

    int dist = std::accumulate(int_dimensions.begin(), int_dimensions.end(), 1, std::multiplies());

    int batch_size = dimensions.size() > rank
                     ? std::accumulate(dimensions.begin() + rank, dimensions.end(), 1, std::multiplies())
                     : 1;

    auto result = cufftPlanMany(&plan, rank, int_dimensions.data(), int_dimensions.data(), 1, dist,
                                int_dimensions.data(), 1, dist,
                                FFT_internal::transform_type<ComplexType>(), batch_size);

    if (result != 0) {
        throw std::runtime_error("FFT plan failed with error " + FFT_internal::CUFFT_error_string(result));
    }
}

template<class ComplexType, class ENABLER>
Gadgetron::cuFFTPlan<ComplexType, ENABLER>::~cuFFTPlan() {
    dpct::fft::fft_engine::destroy(plan);
}

template<class ComplexType, class ENABLER>
void Gadgetron::cuFFTPlan<ComplexType, ENABLER>::fft(cuNDArray<ComplexType> &in_out, bool scale) {
    if (!FFT_internal::compatible_dimensions(rank, in_out.dimensions(), dimensions))
        throw std::runtime_error("Dimensions do not match FFT plan");
    auto result =
        FFT_internal::executePlan(plan, in_out.data(), in_out.data(), -1);
    if (result != 0) {
        throw std::runtime_error("FFT failed with error" + FFT_internal::CUFFT_error_string(result));
    }
    if (scale)
        in_out /= sqrt((realType_t<ComplexType>) std::accumulate(in_out.dimensions().begin(),
                                                                 in_out.dimensions().begin() + rank, 1,
                                                                 std::multiplies()));
}

template<class ComplexType, class ENABLER>
void Gadgetron::cuFFTPlan<ComplexType, ENABLER>::ifft(cuNDArray<ComplexType> &in_out, bool scale) {
    if (!FFT_internal::compatible_dimensions(rank, in_out.dimensions(), dimensions))
        throw std::runtime_error("Dimensions do not match FFT plan");
    auto result =
        FFT_internal::executePlan(plan, in_out.data(), in_out.data(), 1);
    if (result != 0) {
        throw std::runtime_error("IFFT failed with error" + FFT_internal::CUFFT_error_string(result));
    }
    if (scale)
        in_out /= sqrt((realType_t<ComplexType>) std::accumulate(in_out.dimensions().begin(),
                                                                 in_out.dimensions().begin() + rank, 1,
                                                                 std::multiplies()));
}


template<class ComplexType, class ENABLER>
void Gadgetron::cuFFTPlan<ComplexType, ENABLER>::fftc(cuNDArray<ComplexType> &in_out, bool scale) {
    FFT_internal::timeswitch(in_out, rank);
    fft(in_out,scale);
    FFT_internal::timeswitch(in_out, rank);
}

template<class ComplexType, class ENABLER>
void Gadgetron::cuFFTPlan<ComplexType, ENABLER>::ifftc(cuNDArray<ComplexType> &in_out, bool scale) {
    FFT_internal::timeswitch(in_out, rank);
    ifft(in_out,scale);
    FFT_internal::timeswitch(in_out, rank);
}
