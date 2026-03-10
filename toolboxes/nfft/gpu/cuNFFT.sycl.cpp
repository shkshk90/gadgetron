/*
  CUDA implementation of the NFFT.

  -----------

  Accelerating the Non-equispaced Fast Fourier Transform on Commodity Graphics Hardware.
  T.S. Sørensen, T. Schaeffter, K.Ø. Noe, M.S. Hansen. 
  IEEE Transactions on Medical Imaging 2008; 27(4):538-547.

  Real-time Reconstruction of Sensitivity Encoded Radial Magnetic Resonance Imaging Using a Graphics Processing Unit.
  T.S. Sørensen, D. Atkinson, T. Schaeffter, M.S. Hansen.
  IEEE Transactions on Medical Imaging 2009; 28(12): 1974-1985. 
*/

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNFFT.h"

#include "cuNDArray_elemwise.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_utils.h"
#include "cuNDFFT.h"

#include "NFFT.hpp"
#include <cmath>

using namespace Gadgetron;


template<class REAL, unsigned int D, ConvolutionType CONV>
Gadgetron::cuNFFT_impl<REAL, D, CONV>::cuNFFT_impl(
    const vector_td<size_t, D>& matrix_size,
    const vector_td<size_t, D>& matrix_size_os,
    REAL W,
    int device)
  : NFFT_plan<cuNDArray, REAL, D>(matrix_size, matrix_size_os, W)
{
    // Initialize gridding convolution. This was done in base class already, but
    // we need to do it again in order to provide the appropriate convolution
    // type.
    KaiserKernel<REAL, D> kernel(vector_td<unsigned int, D>(matrix_size),
                                 vector_td<unsigned int, D>(matrix_size_os),
                                 W);
    this->conv_ = GriddingConvolution<cuNDArray, complext<REAL>, D, KaiserKernel>::make(
        matrix_size, matrix_size_os, kernel, CONV);

    // Minimal initialization.
    this->initialize(device);
}


template<class REAL, unsigned int D, ConvolutionType CONV>
Gadgetron::cuNFFT_impl<REAL, D, CONV>::~cuNFFT_impl()
{

}


template<class REAL, unsigned int D, ConvolutionType CONV>
void
Gadgetron::cuNFFT_impl<REAL, D, CONV>::fft(cuNDArray <complext<REAL>>& data, NFFT_fft_mode mode, bool do_scale) {

    typename uint64d<D>::Type _dims_to_transform = counting_vec<size_t, D>();
    std::vector<size_t> dims_to_transform = to_std_vector(_dims_to_transform);

    if (mode == NFFT_fft_mode::FORWARDS) {
        cuNDFFT<REAL>::instance()->fft(&data, &dims_to_transform, do_scale);
    } else {
        cuNDFFT<REAL>::instance()->ifft(&data, &dims_to_transform, do_scale);
    }

}


template<class REAL, unsigned int D, ConvolutionType CONV>
void
Gadgetron::cuNFFT_impl<REAL, D, CONV>::deapodize(cuNDArray <complext<REAL>>& image, bool fourier_domain) {

    typename uint64d<D>::Type image_dims = from_std_vector<size_t, D>(image.get_dimensions());
    bool oversampled_image = (image_dims == this->matrix_size_os_);

    if (!oversampled_image) {
        throw std::runtime_error("Error: cuNFFT_impl::deapodize: ERROR: oversampled image not provided as input.");
    }
    if (fourier_domain) {
        if (!deapodization_filterFFT)
            deapodization_filterFFT = compute_deapodization_filter(true);
        image *= *deapodization_filterFFT;
    } else {
        if (!deapodization_filter)
            deapodization_filter = compute_deapodization_filter(false);
        image *= *deapodization_filter;
    }
}

template <class REAL, unsigned int D, ConvolutionType CONV>
void Gadgetron::cuNFFT_impl<REAL, D, CONV>::initialize(int device) try {
    // Device checks.
    if (DPCT_CHECK_ERROR(this->device_ = dpct::get_current_device_id()) != 0)
    {
        throw cuda_error("Error: cuNFFT_impl::barebones:: unable to get this->device_ no");
    }
    
    if (device < 0)
    {
        if (DPCT_CHECK_ERROR(this->device_ = dpct::get_current_device_id()) != 0)
        {
            throw cuda_error("Error: cuNFFT_impl::setup: unable to determine "
                             "device properties.");
        }
    }
    else
    {
        this->device_ = device;
    }

    int device_no_old;
    if (DPCT_CHECK_ERROR(device_no_old = dpct::get_current_device_id()) != 0)
        throw cuda_error("Error: cuNFFT_impl::setup: unable to get device number");

    if (this->device_ != device_no_old &&
        /*
        DPCT1093:34: The "this->device_" device may be not the one intended for use. Adjust the selected
           device if needed.
        */
        DPCT_CHECK_ERROR(dpct::select_device(this->device_)) != 0)
        throw cuda_error("Error: cuNFFT_impl::setup: unable to set device");

    if (this->device_ != device_no_old &&
        /*
        DPCT1093:35: The "device_no_old" device may be not the one intended for use. Adjust the selected
           device if needed.
        */
        DPCT_CHECK_ERROR(dpct::select_device(device_no_old)) != 0)
        throw cuda_error("Error: cuNFFT_impl::setup: unable to restore device");
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//
// Grid fictitious trajectory with a single sample at the origin
//

template<class REAL, unsigned int D, template<class, unsigned int> class K>
void
compute_deapodization_filter_kernel(typename uintd<D>::Type matrix_size_os,
                                    typename reald<REAL, D>::Type matrix_size_os_real,
                                    complext <REAL> *__restrict__ image_os,
                                    const ConvolutionKernel<REAL, D, K>* kernel)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const unsigned int num_elements = prod(matrix_size_os);

    if (idx < num_elements) {

        // Compute weight from Kaiser-Bessel filter
        const typename uintd<D>::Type cell_pos = idx_to_co(idx, matrix_size_os);

        // Sample position ("origin")
        const vector_td<REAL, D> sample_pos = REAL(0.5) * matrix_size_os_real;

        // Calculate the distance between the cell and the sample
        vector_td<REAL, D>
        cell_pos_real = vector_td<REAL, D>(cell_pos);
        const typename reald<REAL, D>::Type delta = sycl::fabs(sample_pos - cell_pos_real);

        // Compute convolution weight.
        REAL weight = kernel->get(delta);

        // Output weight
        image_os[idx] = complext<REAL>(weight, 0.0f);
    }
}

//
// Function to calculate the deapodization filter
//

template<class REAL, unsigned int D, ConvolutionType CONV>
boost::shared_ptr<cuNDArray < complext < REAL> > >
Gadgetron::cuNFFT_impl<REAL, D, CONV>::compute_deapodization_filter(bool FFTed) {
    std::vector<size_t> tmp_vec_os = to_std_vector(this->matrix_size_os_);

    auto  filter = boost::make_shared<cuNDArray<complext<REAL>>>(tmp_vec_os);
    vector_td<REAL, D>
    matrix_size_os_real = vector_td<REAL, D>(this->matrix_size_os_);

    // Find dimensions of grid/blocks.
    dpct::dim3 dimBlock(256);
    dpct::dim3 dimGrid((prod(this->matrix_size_os_) + dimBlock.x - 1) / dimBlock.x);

    // Invoke kernel
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit,
     * query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    /*
    DPCT1129:0: The type "vector_td<unsigned int, D>" is used in the SYCL kernel, but it is not device copyable.
     * The sycl::is_device_copyable specialization has been added for this type. Please review the code.
    */
    /*
    DPCT1129:1: The type "vector_td<REAL, D>" is used in the SYCL kernel, but it is not device copyable. The
     * sycl::is_device_copyable specialization has been added for this type. Please review the code.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto vector_td_unsigned_int_D_this_matrix_size_os__ct0 = vector_td<unsigned int, D>(this->matrix_size_os_);
            auto filter_get_data_ptr_ct2 = filter->get_data_ptr();
          auto this_conv__get_get_kernel_d_ct3 = this->conv_.get())->get_kernel_d();

            cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

            /*
          DPCT1050:36: The template argument of the dpct_kernel_name could not be deduced. You need to
             * update this code.
          */
            cgh.parallel_for<dpct_kernel_name<class compute_deapodization_filter_kernel_f5514e, REAL,
                                              dpct_kernel_scalar<D>, dpct_placeholder /*Fix the type mannually*/>>(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), exp_props, [=](sycl::nd_item<3> item_ct1) {
                    compute_deapodization_filter_kernel<REAL, D>(vector_td_unsigned_int_D_this_matrix_size_os__ct0,
                                                                 matrix_size_os_real, filter_get_data_ptr_ct2,
                                                                 this_conv__get_get_kernel_d_ct3);
                });
        });
    }

    CHECK_FOR_CUDA_ERROR();

    // FFT
    if (FFTed) {
        fft(*filter, NFFT_fft_mode::FORWARDS, false);
    } else {
        fft(*filter, NFFT_fft_mode::BACKWARDS, false);
    }
    // Reciprocal
    reciprocal_inplace(filter.get());
    return filter;
}


namespace Gadgetron {
    template<class REAL, unsigned int D>
    boost::shared_ptr<cuNFFT_plan<REAL, D>> NFFT<cuNDArray, REAL, D>::make_plan(const vector_td<size_t,D>& matrix_size, const vector_td<size_t,D>& matrix_size_os, REAL W, ConvolutionType conv) {
        switch (conv) {
            case ConvolutionType::STANDARD:
                return boost::make_shared<cuNFFT_impl<REAL, D, ConvolutionType::STANDARD> >(matrix_size,matrix_size_os,W);
            case ConvolutionType::ATOMIC:
                return boost::make_shared<cuNFFT_impl<REAL, D, ConvolutionType::ATOMIC> >(matrix_size,matrix_size_os,W);
            case ConvolutionType::SPARSE_MATRIX:
                return boost::make_shared<cuNFFT_impl<REAL, D, ConvolutionType::SPARSE_MATRIX>>(matrix_size,matrix_size_os,W);
        }
        throw std::runtime_error(
                "Invalid convolution type provided. If you're reading this, you may have broken your computer quite badly");
    }

    template<unsigned int D>

    boost::shared_ptr<cuNFFT_plan<double, D>> NFFT<cuNDArray, double, D>::make_plan(const vector_td<size_t,D>& matrix_size, const vector_td<size_t,D>& matrix_size_os, double W, ConvolutionType conv) {
        if (conv == ConvolutionType::STANDARD) {
            return boost::make_shared<cuNFFT_impl<double, D, ConvolutionType::STANDARD>>(matrix_size,matrix_size_os,W);
        }
        throw std::runtime_error("Only standard convolution type supported for doubles");
    }
}

//
// Template instantion
//

template
class Gadgetron::cuNFFT_impl<float, 1, ConvolutionType::ATOMIC>;


template
class Gadgetron::cuNFFT_impl<float, 1, ConvolutionType::SPARSE_MATRIX>;

template
class Gadgetron::cuNFFT_impl<float, 1>;

template
class Gadgetron::cuNFFT_impl<double, 1>;

template
class Gadgetron::cuNFFT_impl<float, 2, ConvolutionType::ATOMIC>;


template
class Gadgetron::cuNFFT_impl<float, 2, ConvolutionType::SPARSE_MATRIX>;

template
class Gadgetron::cuNFFT_impl<float, 2>;

template
class Gadgetron::cuNFFT_impl<double, 2>;

template
class Gadgetron::cuNFFT_impl<float, 3, ConvolutionType::ATOMIC>;

template
class Gadgetron::cuNFFT_impl<float, 3, ConvolutionType::SPARSE_MATRIX>;

template
class Gadgetron::cuNFFT_impl<float, 3>;

template
class Gadgetron::cuNFFT_impl<double, 3>;

template
class Gadgetron::cuNFFT_impl<float, 4, ConvolutionType::ATOMIC>;


template
class Gadgetron::cuNFFT_impl<float, 4, ConvolutionType::SPARSE_MATRIX>;
template
class Gadgetron::cuNFFT_impl<float, 4>;

template
class Gadgetron::cuNFFT_impl<double, 4>;


template class Gadgetron::NFFT<cuNDArray,float,1>;
template class Gadgetron::NFFT<cuNDArray,float,2>;
template class Gadgetron::NFFT<cuNDArray,float,3>;
template class Gadgetron::NFFT<cuNDArray,float,4>;



template class Gadgetron::NFFT<cuNDArray,double,1>;
template class Gadgetron::NFFT<cuNDArray,double,2>;
template class Gadgetron::NFFT<cuNDArray,double,3>;
template class Gadgetron::NFFT<cuNDArray,double,4>;
