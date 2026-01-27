#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDDWT.h"
#include "vector_td.h"
#include "vector_td_utilities.h"
#include "complext.h"
#include "cuNDArray_math.h"
#include <cmath>

using namespace Gadgetron;


template<class T, unsigned int D> struct Daubechies {

};
/*
template <class T> struct Daubechies<T,4>{
	static vector_td<T,4> coefficients = vector_td<T,4>({0.6830127,1.1830127,0.3169873,-0.1830127});
	//vector_td<T,4> coefficients;
};
 */

/* DPCT_ORIG template<class T, unsigned int D, unsigned int WD> __global__ static void
dwt_kernel( vector_td<int,D> dims,  const T  * __restrict__ in, T * __restrict__ out, int dir, vector_td<typename
realType<T>::Type,WD> wavelet, int shift )*/
template <class T, unsigned int D, unsigned int WD>
static void dwt_kernel(vector_td<int, D> dims, const T* __restrict__ in, T* __restrict__ out, int dir,
                       vector_td<typename realType<T>::Type, WD> wavelet, int shift)
{
/* DPCT_ORIG 	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if( idx < prod(dims)/2){
		vector_td<int,D> dims2 = dims;
		dims2[dir] /= 2;
		vector_td<T,WD> data;
		typename intd<D>::Type co = idx_to_co(idx, dims2);
		//co[dir] *= 2; //We're doing the decimated wavelet
		co[dir] = (co[dir]+shift+dims[dir])%dims[dir]; //Wrap around
		for (int i = 0; i < WD; i++){
			data[i] = in[co_to_idx(co, dims)];
			co[dir] = (co[dir]+1+dims[dir])%dims[dir]; //Wrap around
		}
		T s = dot(data,wavelet); //Getting the scaling element is easy

		T d = 0;
		float sign = 1;

		//Reverse wavelet and shift sign of every second element
		for (int i = 0; i< WD; i++){
			d+= wavelet[WD-i-1]*sign*data[i];
			sign *= -1;
		}

		//co = idx_to_co(idx,dims2);
		//size_t out_index = co_to_idx(co,dims2);
		out[idx] = s;
		out[idx+prod(dims)/2] =d;
	}
}

/* DPCT_ORIG template<class T, unsigned int D, unsigned int WD> __global__ static void
idwt_kernel( vector_td<int,D> dims,  const T  * __restrict__ in, T * __restrict__ out, int dim, vector_td<typename
realType<T>::Type,WD> wavelet, int shift )*/
template <class T, unsigned int D, unsigned int WD>
static void idwt_kernel(vector_td<int, D> dims, const T* __restrict__ in, T* __restrict__ out, int dim,
                        vector_td<typename realType<T>::Type, WD> wavelet, int shift)
{
/* DPCT_ORIG 	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if( idx < prod(dims)/2){
		vector_td<int,D> dims2 = dims;
		dims2[dim] /= 2;
		typename intd<D>::Type co = idx_to_co(idx, dims2);


		T res1 = 0;
		T res2 = 0;
		co[dim] = (co[dim]+dims2[dim]+WD-1)%dims2[dim];
		for (int i = 0; i < WD/2; i++){
			T s = in[co_to_idx(co,dims2)];
			res1 += wavelet[2*i]*s;
			res2 += wavelet[2*i+1]*s;
			co[dim] = (co[dim]-1+dims2[dim])%dims2[dim];
		}

		//Create the diff coefficients. Yes we could compute them on the fly.
		vector_td<T,WD> diff;
		{
			float sign = 1;
			for (int i = 0; i < WD; i++){
				diff[i] = wavelet[WD-i-1]*sign;
				sign *= -1;
			}
		}

		co = idx_to_co(idx, dims2);
		//co[dim] += dims2[dim];
		//co[dim] = (co[dim]+dims[dim]+WD-1)%dims[dim];

		co[dim] = (co[dim]+dims2[dim]+WD-1)%dims2[dim];
		//co[dim] += dims2[dim];
		for (int i = 0; i < WD/2; i++){
			T d = in[co_to_idx(co,dims2)+prod(dims)/2];
			res1 += diff[2*i]*d;
			res2 += diff[2*i+1]*d;
			co[dim] = (co[dim]-1+dims2[dim])%dims2[dim];
		}

		co = idx_to_co(idx, dims2);
		co[dim] *= 2;
		co[dim] = (co[dim]+dims[dim]+shift+2*WD-2)%dims[dim];
		out[co_to_idx(co,dims)] = res1;
		co[dim] = (co[dim]+dims[dim]+1)%dims[dim];
		out[co_to_idx(co,dims)] = res2;
	}
}
static inline bool isPowerOfTwo (size_t x)
{
	return ((x != 0) && ((x & (~x + 1)) == x));
}


/*
const

template < struct Daubechies {
	({0.6830127,1.1830127,0.3169873,-0.1830127});
};
/*
template<class T, unsigned int D, class wave> __device__ static void lift(T& data[D]){

}
 */
/**
 *
 * @param in Input array
 * @param out Output array
 * @param wavelet vector of the scaling function coefficients for the wavelet
 */
template<class T, unsigned int D, unsigned int WD> void Gadgetron::DWT1( cuNDArray<T>* in, cuNDArray<T>* out, vector_td<typename realType<T>::Type,WD> wavelet, int dim, int shift){

	if (!(isPowerOfTwo(in->get_size(dim)) && in->get_size(dim) >= WD)){
		GINFO(std::string("Dimension is: " + std::to_string(in->get_size(dim)) + "\n").c_str());
		throw std::runtime_error("DWT: Illegal input dimensions for DWT. Power of two reconstructions only");
	}

	size_t tot_threads = in->get_number_of_elements()/2; //1 thread per 2 elements
	int threadsPerBlock =std::min(tot_threads,size_t(256));
/* DPCT_ORIG 	dim3 dimBlock( threadsPerBlock);*/
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = std::max(size_t(1),tot_threads/threadsPerBlock);
/* DPCT_ORIG 	dim3 dimGrid(totalBlocksPerGrid);*/
        dpct::dim3 dimGrid(totalBlocksPerGrid);

        const typename intd<D>::Type dims = vector_td<int,D>( from_std_vector<size_t,D>(in->get_dimensions()));
/* DPCT_ORIG 	dwt_kernel<T,D,WD><<<dimGrid,dimBlock>>>(dims,
 * in->get_data_ptr(),out->get_data_ptr(),dim,wavelet,shift);*/
        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    /*
    DPCT1129:28: The type "vector_td<typename realType<T>::Type, WD>" is used in the SYCL kernel, but it is not
    device copyable. The sycl::is_device_copyable specialization has been added for this type. Please review the
    code.
    */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto in_get_data_ptr_ct1 = in->get_data_ptr();
            auto out_get_data_ptr_ct2 = out->get_data_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class dwt_kernel_443a8f, T, dpct_kernel_scalar<D>, dpct_kernel_scalar<WD>>>(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                    dwt_kernel<T, D, WD>(dims, in_get_data_ptr_ct1, out_get_data_ptr_ct2, dim, wavelet, shift);
                });
        });
    }
/* DPCT_ORIG 	cudaDeviceSynchronize();*/
        dpct::get_current_device().queues_wait_and_throw();
        CHECK_FOR_CUDA_ERROR()
	*out *= T(1.0/std::sqrt(sum(wavelet)));


}

/**
 *
 * @param in Input array
 * @param out Output array
 * @param wavelet vector of the scaling function coefficients for the wavelet
 */
template<class T, unsigned int D, unsigned int WD> void Gadgetron::IDWT1( cuNDArray<T>* in, cuNDArray<T>* out, vector_td<typename realType<T>::Type,WD> wavelet, int dim, int shift){

	if (!(isPowerOfTwo(in->get_size(dim)) && in->get_size(dim) >= WD)){
		GINFO(std::string(
			"Dimension " +
			std::to_string(dim) +
			" is: " +
			std::to_string(in->get_size(dim)) +
			" " +
			std::to_string(in->get_number_of_dimensions()) +
			"\n").c_str());
		throw std::runtime_error("IDWT: Illegal input dimensions for DWT. Power of two reconstructions only");
	}

	size_t tot_threads = in->get_number_of_elements()/2; //1 thread per 2 elements
	int threadsPerBlock =std::min(tot_threads,size_t(256));
/* DPCT_ORIG 	dim3 dimBlock( threadsPerBlock);*/
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = std::max(size_t(1),tot_threads/threadsPerBlock);
/* DPCT_ORIG 	dim3 dimGrid(totalBlocksPerGrid);*/
        dpct::dim3 dimGrid(totalBlocksPerGrid);

        const typename intd<D>::Type dims = vector_td<int,D>( from_std_vector<size_t,D>(in->get_dimensions()));
/* DPCT_ORIG 	idwt_kernel<T,D,WD><<<dimGrid,dimBlock>>>(dims,
 * in->get_data_ptr(),out->get_data_ptr(),dim,wavelet,shift);*/
        /*
        DPCT1049:31: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    /*
    DPCT1129:30: The type "vector_td<typename realType<T>::Type, WD>" is used in the SYCL kernel, but it is not
    device copyable. The sycl::is_device_copyable specialization has been added for this type. Please review the
    code.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
        auto in_get_data_ptr_ct1 = in->get_data_ptr();
        auto out_get_data_ptr_ct2 = out->get_data_ptr();

        cgh.parallel_for<dpct_kernel_name<class idwt_kernel_70206e, T, dpct_kernel_scalar<D>, dpct_kernel_scalar<WD>>>(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                idwt_kernel<T, D, WD>(dims, in_get_data_ptr_ct1, out_get_data_ptr_ct2, dim, wavelet, shift);
            });
    });

        *out *= T(1.0/std::sqrt(sum(wavelet)));
	CHECK_FOR_CUDA_ERROR();


}

template EXPORTGPUDWT void Gadgetron::DWT1<float,2,6>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,6> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float,2,6>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,6> ,int,int);
template EXPORTGPUDWT void Gadgetron::DWT1<float,2,4>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,4> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float,2,4>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,4> ,int,int);
template EXPORTGPUDWT void Gadgetron::DWT1<float,2,2>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,2> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float,2,2>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,2> ,int,int);

template EXPORTGPUDWT void Gadgetron::DWT1<float_complext,2,6>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,6> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float_complext,2,6>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,6> ,int,int);
template EXPORTGPUDWT void Gadgetron::DWT1<float_complext,2,4>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,4> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float_complext,2,4>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,4> ,int,int);
template EXPORTGPUDWT void Gadgetron::DWT1<float_complext,2,2>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,2> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float_complext,2,2>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,2> ,int,int);


template EXPORTGPUDWT void Gadgetron::DWT1<float,3,6>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,6> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float,3,6>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,6> ,int,int);
template EXPORTGPUDWT void Gadgetron::DWT1<float,3,4>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,4> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float,3,4>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,4> ,int,int);
template EXPORTGPUDWT void Gadgetron::DWT1<float,3,2>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,2> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float,3,2>(cuNDArray<float>*, cuNDArray<float>*, vector_td<float,2> ,int,int);

template EXPORTGPUDWT void Gadgetron::DWT1<float_complext,3,6>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,6> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float_complext,3,6>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,6> ,int,int);
template EXPORTGPUDWT void Gadgetron::DWT1<float_complext,3,4>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,4> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float_complext,3,4>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,4> ,int,int);
template EXPORTGPUDWT void Gadgetron::DWT1<float_complext,3,2>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,2> ,int,int);
template EXPORTGPUDWT void Gadgetron::IDWT1<float_complext,3,2>(cuNDArray<float_complext>*, cuNDArray<float_complext>*, vector_td<float,2> ,int,int);
