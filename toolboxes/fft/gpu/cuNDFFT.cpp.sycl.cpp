#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDFFT.h"
#include "vector_td.h"
#include "cuNDArray.h"
#include "cuNDArray_utils.h"
#include "cuNDArray_operators.h"
#include <dpct/fft_utils.hpp>

#include <sstream>
#include <complex>
#include <iostream>

namespace Gadgetron{

// RAII guard to ensure FFT plans are always destroyed
struct FFTPlanGuard {
    dpct::fft::fft_engine_ptr plan = nullptr;
    FFTPlanGuard() = default;
    FFTPlanGuard(const FFTPlanGuard&) = delete;
    FFTPlanGuard& operator=(const FFTPlanGuard&) = delete;
    ~FFTPlanGuard() {
        if (plan) {
            try { dpct::fft::fft_engine::destroy(plan); } catch (...) {}
            plan = nullptr;
        }
    }
};

template<class T> cuNDFFT<T>* cuNDFFT<T>::instance()
  				{
	if (!__instance)
		__instance = new cuNDFFT<T>;
	return __instance;
  				}

template<class T> cuNDFFT<T>* cuNDFFT<T>::__instance = NULL;

template <class T> dpct::fft::fft_type get_transform_type();
template <>
dpct::fft::fft_type get_transform_type<float>() {
    return dpct::fft::fft_type::complex_float_to_complex_float;
}
template <>
dpct::fft::fft_type get_transform_type<double>() {
    return dpct::fft::fft_type::complex_double_to_complex_double;
}

template <class T>
int cuNDA_FFT_execute(dpct::fft::fft_engine_ptr plan,
                      cuNDArray<complext<T>> *in_out, int direction);

template <>
int cuNDA_FFT_execute<float>(dpct::fft::fft_engine_ptr plan,
                             cuNDArray<float_complext> *in_out, int direction) {
        return DPCT_CHECK_ERROR((plan->compute<sycl::float2, sycl::float2>((sycl::float2*)in_out->get_data_ptr(), (sycl::float2*)in_out->get_data_ptr(), direction == 1 ? dpct::fft::fft_direction::backward : dpct::fft::fft_direction::forward))); }

template <>
int cuNDA_FFT_execute<double>(dpct::fft::fft_engine_ptr plan,
                              cuNDArray<double_complext> *in_out,
                              int direction) {
        return DPCT_CHECK_ERROR((plan->compute<sycl::double2, sycl::double2>((sycl::double2*)in_out->get_data_ptr(), (sycl::double2*)in_out->get_data_ptr(), direction == 1 ? dpct::fft::fft_direction::backward : dpct::fft::fft_direction::forward))); }

template <class T>
void cuNDFFT<T>::fft_int(cuNDArray<complext<T>> *input,
                         std::vector<size_t> *dims_to_transform, int direction,
                         bool do_scale) try {
        std::vector<size_t> new_dim_order;
	std::vector<size_t> reverse_dim_order;
	std::vector<size_t> dim_count(input->get_number_of_dimensions(),0);

	size_t array_ndim = input->get_number_of_dimensions();
	std::vector<size_t> array_dims = input->get_dimensions();

	std::vector<size_t> dims = std::vector<size_t>(dims_to_transform->size(),0);
	for (size_t i = 0; i < dims_to_transform->size(); i++) {
		if ((*dims_to_transform)[i] >= array_ndim) {
			std::stringstream ss;
			ss << "cuNDFFT::fft Invalid dimensions specified for transform " << (*dims_to_transform)[i] << "max " << array_ndim;
			throw std::runtime_error(ss.str());;
		}
		if (dim_count[(*dims_to_transform)[i]] > 0) {
			throw std::runtime_error("cuNDFFT::fft Invalid dimensions (duplicates) specified for transform");;
		}
		dim_count[(*dims_to_transform)[i]]++;
		dims[dims_to_transform->size()-1-i] = array_dims[(*dims_to_transform)[i]];
	}

	new_dim_order = *dims_to_transform;
	for (size_t i = 0; i < array_ndim; i++) {
		if (!dim_count[i]) new_dim_order.push_back(i);
	}

	size_t ndim = dims.size();
	bool must_permute = false;

	{
		for (size_t i = 0; i < new_dim_order.size(); i++)
			must_permute |= (i != new_dim_order[i]);
	}

	//Check if we can use the fast FFT versions instead.
	if (!must_permute){
		switch (ndim){
		case 1:
			fft1_int(input,direction,do_scale);
			return;
			break;
		case 2:
			fft2_int(input,direction,do_scale);
			return;
			break;
		case 3:
			fft3_int(input,direction,do_scale);
			return;
			break;
		default:
			break;
		}
	}

	reverse_dim_order = std::vector<size_t>(array_ndim,0);
	for (size_t i = 0; i < array_ndim; i++) {
		reverse_dim_order[new_dim_order[i]] = i;
	}


	size_t batches = 0;
	size_t elements_in_ft = 1;
	for (size_t i = 0; i < dims.size(); i++)
		elements_in_ft *= dims[i];
	batches = input->get_number_of_elements() / elements_in_ft;

        FFTPlanGuard guard;
        int ftres;

        std::vector<int> int_dims;
	for( unsigned int i=0; i<dims.size(); i++ )
		int_dims.push_back((int)dims[i]);

        ftres = DPCT_CHECK_ERROR(
            guard.plan = dpct::fft::fft_engine::create(
                &dpct::get_in_order_queue(), ndim, &int_dims[0], &int_dims[0],
                1, elements_in_ft, &int_dims[0], 1, elements_in_ft,
                get_transform_type<T>(), batches));
        if (ftres != 0) {
		std::cerr << "cuNDFFT::fft_int plan creation failed: ftres=" << ftres
		          << " ndim=" << ndim << " batches=" << batches
		          << " elements_in_ft=" << elements_in_ft
		          << " direction=" << direction << " dims=[";
		for (size_t i = 0; i < int_dims.size(); i++)
			std::cerr << (i ? "," : "") << int_dims[i];
		std::cerr << "]" << std::endl;
		throw std::runtime_error("cuNDFFT::fft_int FFT plan failed: " + std::to_string(ftres));
	}

	if (must_permute)
		*input = permute(*input,new_dim_order);

		for (size_t i =0; i < dims_to_transform->size(); i++)
			timeswitch(input,dims_to_transform->at(i));

        if (cuNDA_FFT_execute<T>(guard.plan, input, direction) != 0) {
		std::cerr << "cuNDFFT::fft_int execute failed:"
		          << " ndim=" << ndim << " batches=" << batches
		          << " elements_in_ft=" << elements_in_ft
		          << " direction=" << direction << " dims=[";
		for (size_t i = 0; i < int_dims.size(); i++)
			std::cerr << (i ? "," : "") << int_dims[i];
		std::cerr << "]" << std::endl;
		throw std::runtime_error("cuNDFFT::fft_int FFT execute failed");
	}

		for (size_t i =0; i < dims_to_transform->size(); i++)
			timeswitch(input,dims_to_transform->at(i));

	if (do_scale) {
		*input *= 1/std::sqrt(T(elements_in_ft));
	}

	if (must_permute)
		*input = permute(*input,reverse_dim_order);
}
catch (sycl::exception const &exc) {
  std::cerr << "cuNDFFT::fft_int SYCL exception: " << exc.what()
            << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl;
  std::exit(1);
}

template <class T>
void cuNDFFT<T>::fft1_int(cuNDArray<complext<T>> *input, int direction,
                          bool do_scale) try {
        FFTPlanGuard guard;
        int ftres;

        std::vector<int> int_dims {int(input->get_size(0))};
	int elements_in_ft = input->get_size(0);
	int batches = input->get_number_of_elements()/elements_in_ft;
        ftres = DPCT_CHECK_ERROR(
            guard.plan = dpct::fft::fft_engine::create(
                &dpct::get_in_order_queue(), 1, &int_dims[0], &int_dims[0], 1,
                elements_in_ft, &int_dims[0], 1, elements_in_ft,
                get_transform_type<T>(), batches));
        if (ftres != 0) {
		std::cerr << "cuNDFFT::fft1_int plan creation failed: ftres=" << ftres
		          << " dim0=" << int_dims[0] << " batches=" << batches
		          << " direction=" << direction << std::endl;
		throw std::runtime_error("cuNDFFT::fft1_int FFT plan failed: " + std::to_string(ftres));
	}

		timeswitch1D(input);

        if (cuNDA_FFT_execute<T>(guard.plan, input, direction) != 0) {
		std::cerr << "cuNDFFT::fft1_int execute failed:"
		          << " dim0=" << int_dims[0] << " batches=" << batches
		          << " direction=" << direction << std::endl;
		throw std::runtime_error("cuNDFFT::fft1_int FFT execute failed");
	}

		timeswitch1D(input);
	if (do_scale) {
		*input *= 1/std::sqrt(T(elements_in_ft));
	}
}
catch (sycl::exception const &exc) {
  std::cerr << "cuNDFFT::fft1_int SYCL exception: " << exc.what()
            << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl;
  std::exit(1);
}

template <class T>
void cuNDFFT<T>::fft2_int(cuNDArray<complext<T>> *input, int direction,
                          bool do_scale) try {
        FFTPlanGuard guard;
        int ftres;

        std::vector<int> int_dims {int(input->get_size(1)),int(input->get_size(0))};
	int elements_in_ft = input->get_size(0)*input->get_size(1);
	int batches = input->get_number_of_elements()/elements_in_ft;
        ftres = DPCT_CHECK_ERROR(
            guard.plan = dpct::fft::fft_engine::create(
                &dpct::get_in_order_queue(), 2, &int_dims[0], &int_dims[0], 1,
                elements_in_ft, &int_dims[0], 1, elements_in_ft,
                get_transform_type<T>(), batches));
        if (ftres != 0) {
		std::cerr << "cuNDFFT::fft2_int plan creation failed: ftres=" << ftres
		          << " dims=[" << int_dims[0] << "," << int_dims[1] << "]"
		          << " batches=" << batches
		          << " elements_in_ft=" << elements_in_ft
		          << " direction=" << direction << std::endl;
		throw std::runtime_error("cuNDFFT::fft2_int FFT plan failed: " + std::to_string(ftres));
	}

		timeswitch2D(input);

        if (cuNDA_FFT_execute<T>(guard.plan, input, direction) != 0) {
		std::cerr << "cuNDFFT::fft2_int execute failed:"
		          << " dims=[" << int_dims[0] << "," << int_dims[1] << "]"
		          << " batches=" << batches
		          << " elements_in_ft=" << elements_in_ft
		          << " direction=" << direction << std::endl;
		throw std::runtime_error("cuNDFFT::fft2_int FFT execute failed");
	}

		timeswitch2D(input);
	if (do_scale) {
		*input *= 1/std::sqrt(T(elements_in_ft));
	}
}
catch (sycl::exception const &exc) {
  std::cerr << "cuNDFFT::fft2_int SYCL exception: " << exc.what()
            << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl;
  std::exit(1);
}
template <class T>
void cuNDFFT<T>::fft3_int(cuNDArray<complext<T>> *input, int direction,
                          bool do_scale) try {
        FFTPlanGuard guard;
        int ftres;

        std::vector<int> int_dims {int(input->get_size(2)),int(input->get_size(1)),int(input->get_size(0))};
	int elements_in_ft = input->get_size(0)*input->get_size(1)*input->get_size(2);
	int batches = input->get_number_of_elements()/elements_in_ft;
        ftres = DPCT_CHECK_ERROR(
            guard.plan = dpct::fft::fft_engine::create(
                &dpct::get_in_order_queue(), 3, &int_dims[0], &int_dims[0], 1,
                elements_in_ft, &int_dims[0], 1, elements_in_ft,
                get_transform_type<T>(), batches));
        if (ftres != 0) {
		std::cerr << "cuNDFFT::fft3_int plan creation failed: ftres=" << ftres
		          << " dims=[" << int_dims[0] << "," << int_dims[1] << "," << int_dims[2] << "]"
		          << " batches=" << batches
		          << " elements_in_ft=" << elements_in_ft
		          << " direction=" << direction << std::endl;
		throw std::runtime_error("cuNDFFT::fft3_int FFT plan failed: " + std::to_string(ftres));
	}

		timeswitch3D(input);

        if (cuNDA_FFT_execute<T>(guard.plan, input, direction) != 0) {
		std::cerr << "cuNDFFT::fft3_int execute failed:"
		          << " dims=[" << int_dims[0] << "," << int_dims[1] << "," << int_dims[2] << "]"
		          << " batches=" << batches
		          << " elements_in_ft=" << elements_in_ft
		          << " direction=" << direction << std::endl;
		throw std::runtime_error("cuNDFFT::fft3_int FFT execute failed");
	}

		timeswitch3D(input);

	if (do_scale) {
		*input *= 1/std::sqrt(T(elements_in_ft));
	}
}
catch (sycl::exception const &exc) {
  std::cerr << "cuNDFFT::fft3_int SYCL exception: " << exc.what()
            << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl;
  std::exit(1);
}
template<class T> void
cuNDFFT<T>::fft( cuNDArray< complext<T> > *input, std::vector<size_t> *dims_to_transform, bool do_scale )
{
        fft_int(input, dims_to_transform, -1, do_scale);
}

template<class T> void
cuNDFFT<T>::ifft( cuNDArray< complext<T> > *input, std::vector<size_t> *dims_to_transform, bool do_scale )
{
        fft_int(input, dims_to_transform, 1, do_scale);
}

template<class T> void
cuNDFFT<T>::fft( cuNDArray< complext<T> > *input, unsigned int dim_to_transform, bool do_scale )
{
	std::vector<size_t> dims(1,dim_to_transform);
        fft_int(input, &dims, -1, do_scale);
}

template<class T> void
cuNDFFT<T>::ifft( cuNDArray< complext<T> > *input, unsigned int dim_to_transform, bool do_scale )
{
	std::vector<size_t> dims(1,dim_to_transform);
        fft_int(input, &dims, 1, do_scale);
}

template<class T> void
cuNDFFT<T>::fft( cuNDArray< complext<T> > *input, bool do_scale )
{
	std::vector<size_t> dims(input->get_number_of_dimensions(),0);
	for (size_t i = 0; i < dims.size(); i++) dims[i] = i;
        fft_int(input, &dims, -1, do_scale);
}

template<class T> void
cuNDFFT<T>::ifft( cuNDArray<complext<T> > *input, bool do_scale )
{
	std::vector<size_t> dims(input->get_number_of_dimensions(),0);
	for (size_t i = 0; i < dims.size(); i++) dims[i] = i;
        fft_int(input, &dims, 1, do_scale);
}

template<class T> void
cuNDFFT<T>::fft1( cuNDArray< complext<T> > *input, bool do_scale )
{
        fft1_int(input, -1, do_scale);
}
template<class T> void
cuNDFFT<T>::fft2( cuNDArray< complext<T> > *input, bool do_scale )
{
        fft2_int(input, -1, do_scale);
}
template<class T> void
cuNDFFT<T>::fft3( cuNDArray< complext<T> > *input, bool do_scale )
{
        fft3_int(input, -1, do_scale);
}

template<class T> void
cuNDFFT<T>::ifft1( cuNDArray< complext<T> > *input, bool do_scale )
{
        fft1_int(input, 1, do_scale);
}
template<class T> void
cuNDFFT<T>::ifft2( cuNDArray< complext<T> > *input, bool do_scale )
{
        fft2_int(input, 1, do_scale);
}
template<class T> void
cuNDFFT<T>::ifft3( cuNDArray< complext<T> > *input, bool do_scale )
{
        fft3_int(input, 1, do_scale);
}
// Instantiation
template class EXPORTGPUFFT cuNDFFT<float>;
template class EXPORTGPUFFT cuNDFFT<double>;
}
