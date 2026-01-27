/*
 * cuPartialDerivativeOperator2.cu
 *
 *  Created on: Jul 27, 2012
 *      Author: David C Hansen
 */

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuPartialDerivativeOperator2.h"
#include "vector_td_utilities.h"
#include "check_CUDA.h"
#include "cuNDArray_math.h"
#include <cmath>

#define MAX_THREADS_PER_BLOCK 512

using namespace Gadgetron;
/* DPCT_ORIG template<class T, unsigned int D> __global__ void
partial_derivative_kernel2_forwards( typename intd<D>::Type dims,
                                        T *in, T *out )*/
template <class T, unsigned int D> void partial_derivative_kernel2_forwards(typename intd<D>::Type dims, T* in, T* out)
{

/* DPCT_ORIG   const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  if( idx < prod(dims) ){

    T valN1, valC;

    const typename intd<D>::Type co = idx_to_co(idx, dims);

    typename intd<D>::Type coN1 = co;
    coN1[D-1] +=1;
	if (co[D-1] == dims[D-1]-1)  coN1[D-1] -= 2;



    valN1 = in[co_to_idx(coN1, dims)];

    valC = in[idx];
    out[idx] += valC-valN1;
  }
}

/* DPCT_ORIG template<class T, unsigned int D> __global__ void
partial_derivative_kernel2_backwards( typename intd<D>::Type dims,
                                        T *in, T *out )*/
template <class T, unsigned int D> void partial_derivative_kernel2_backwards(typename intd<D>::Type dims, T* in, T* out)
{

/* DPCT_ORIG   const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  if( idx < prod(dims) ){

    T valN1, valC;

    const typename intd<D>::Type co = idx_to_co(idx, dims);

    if (co[D-1] == 0) out[idx] += in[idx];
    else {
		typename intd<D>::Type coN1 = co;
		coN1[D-1] -=1;


		valN1 = in[co_to_idx(coN1, dims)];

		valC = in[idx];
		T val = valC-valN1;
		if (co[D-1]== dims[D-1]-2){
			coN1[D-1] += 2;
			val -= in[co_to_idx(coN1, dims)];
		}
		out[idx] += val;
    }
  }
}

/* DPCT_ORIG template<class T, unsigned int D> __global__ void
second_order_partial_derivative_kernel2( typename intd<D>::Type dims,
                                        T *in, T *out )*/
template <class T, unsigned int D>
void second_order_partial_derivative_kernel2(typename intd<D>::Type dims, T* in, T* out)
{
/* DPCT_ORIG   const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
  if( idx < prod(dims) ){

    T valN1, valN2, valC;

    const typename intd<D>::Type co = idx_to_co(idx, dims);

    typename intd<D>::Type coN1 = co;
    coN1[D-1] +=1;
	if (co[D-1] == dims[D-1]-1)  coN1[D-1] -= 3;
    typename intd<D>::Type coN2 = co;
    coN2[D-1] -=1;



    valN1 = in[co_to_idx(coN1, dims)];
    if (co[D-1] == 0)  valN2 = 0;
    else valN2 = in[co_to_idx(coN2, dims)];

    valC = in[idx];
    T val;

    if (co[D-1] < dims[D-1]-2 ) val = valC+valC-valN1-valN2;
    else if (co[D-1] == dims[D-1]-2 ) val =  2*valC+valC-2*valN1-valN2;
    else val =  2*valC-valN1-2*valN2;

    out[idx] += val;
  }
}


template< class T, unsigned int D> void
cuPartialDerivativeOperator2<T,D>::mult_MH_M(cuNDArray<T> *in, cuNDArray<T> *out,
										bool accumulate )
{

	cuNDArray<T> tmp = *in;
	mult_M(in,&tmp,false);
	mult_MH(&tmp,out,accumulate);

}

template< class T, unsigned int D> void
cuPartialDerivativeOperator2<T,D>::mult_MH(cuNDArray<T> *in, cuNDArray<T> *out,
										bool accumulate )
{
  if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
	  throw std::runtime_error( "partialDerivativeOperator2::mult_MH : array dimensions mismatch." );
  }

  if( in->get_number_of_dimensions() != D || out->get_number_of_dimensions() != D ){
	  throw std::runtime_error( "partialDerivativeOperator2::mult_MH  : dimensionality mismatch" );
  }

  typename uintd<D>::Type _dims = vector_td<unsigned int,D>(from_std_vector<size_t,D>( in->get_dimensions()) );
  typename intd<D>::Type dims;
  for( unsigned int i=0; i<D; i++ ){
    dims.vec[i] = (int)_dims.vec[i];
  }



  if (!accumulate) clear(out);
  int threadsPerBlock =std::min(prod(dims),MAX_THREADS_PER_BLOCK);
/* DPCT_ORIG   dim3 dimBlock( threadsPerBlock);*/
  dpct::dim3 dimBlock(threadsPerBlock);
  int totalBlocksPerGrid = (prod(dims)+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK;
/* DPCT_ORIG   dim3 dimGrid(totalBlocksPerGrid);*/
  dpct::dim3 dimGrid(totalBlocksPerGrid);

  // Invoke kernel
/* DPCT_ORIG   partial_derivative_kernel2_backwards<T,D><<< dimGrid, dimBlock >>> ( dims, in->get_data_ptr(),
 * out->get_data_ptr() );*/
  /*
  DPCT1049:94: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the work-group size if needed.
  */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto in_get_data_ptr_ct1 = in->get_data_ptr();
            auto out_get_data_ptr_ct2 = out->get_data_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class partial_derivative_kernel2_backwards_dbc2d5, T, dpct_kernel_scalar<D>>>(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                    partial_derivative_kernel2_backwards<T, D>(dims, in_get_data_ptr_ct1, out_get_data_ptr_ct2);
                });
        });
    }

  CHECK_FOR_CUDA_ERROR();



}

template< class T, unsigned int D> void
cuPartialDerivativeOperator2<T,D>::mult_M(cuNDArray<T> *in, cuNDArray<T> *out,
										bool accumulate )
{
  if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
    throw std::runtime_error( "partialDerivativeOperator2::mult_M : array dimensions mismatch.");

  }

  if( in->get_number_of_dimensions() != D || out->get_number_of_dimensions() != D ){
	  throw std::runtime_error( "partialDerivativeOperator2::mult_M  : dimensionality mismatch" );
  }

  typename uintd<D>::Type _dims = vector_td<unsigned int,D>(from_std_vector<size_t,D>( in->get_dimensions()) );
  typename intd<D>::Type dims;
  for( unsigned int i=0; i<D; i++ ){
    dims.vec[i] = (int)_dims.vec[i];
  }



  if (!accumulate) clear(out);

  int threadsPerBlock =std::min(prod(dims),MAX_THREADS_PER_BLOCK);
/* DPCT_ORIG    dim3 dimBlock( threadsPerBlock);*/
   dpct::dim3 dimBlock(threadsPerBlock);
   int totalBlocksPerGrid = (prod(dims)+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK;
/* DPCT_ORIG    dim3 dimGrid(totalBlocksPerGrid);*/
   dpct::dim3 dimGrid(totalBlocksPerGrid);

  // Invoke kernel
/* DPCT_ORIG   partial_derivative_kernel2_forwards<T,D><<< dimGrid, dimBlock >>> ( dims, in->get_data_ptr(),
 * out->get_data_ptr() );*/
  /*
  DPCT1049:95: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the work-group size if needed.
  */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto in_get_data_ptr_ct1 = in->get_data_ptr();
            auto out_get_data_ptr_ct2 = out->get_data_ptr();

            cgh.parallel_for<
                dpct_kernel_name<class partial_derivative_kernel2_forwards_a22deb, T, dpct_kernel_scalar<D>>>(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                    partial_derivative_kernel2_forwards<T, D>(dims, in_get_data_ptr_ct1, out_get_data_ptr_ct2);
                });
        });
    }

  CHECK_FOR_CUDA_ERROR();



}





template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float,1>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float,2>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float,3>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float,4>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float,5>;

template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double,1>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double,2>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double,3>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double,4>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double,5>;


template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float_complext,1>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float_complext,2>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float_complext,3>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float_complext,4>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<float_complext,5>;


template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double_complext,1>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double_complext,2>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double_complext,3>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double_complext,4>;
template class EXPORTGPUOPERATORS Gadgetron::cuPartialDerivativeOperator2<double_complext,5>;

