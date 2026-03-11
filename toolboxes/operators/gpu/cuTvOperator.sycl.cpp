#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuTvOperator.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "vector_td_utilities.h"
#include "complext.h"
#include <iostream>
#include "check_CUDA.h"
#include "cudaDeviceManager.h"
#include <stdio.h>
#include <cmath>

using namespace Gadgetron;

template<class REAL, class T, unsigned int D> static inline  REAL gradient(const T* __restrict__ in, const vector_td<int,D>& dims, vector_td<int,D>& co)
{
	REAL grad = REAL(0);
	T xi = in[co_to_idx((co+dims)%dims,dims)];
	for (int i = 0; i < D; i++){
		co[i]+=1;
		T dt = in[co_to_idx((co+dims)%dims,dims)];
		grad += norm(xi-dt);
		co[i]-=1;
	}
        return sycl::sqrt(grad);
}


template<class REAL, class T, unsigned int D> static void tvGradient_kernel(const T* __restrict__ in, T* __restrict__ out, const vector_td<int,D> dims,REAL limit,REAL weight)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if( idx < prod(dims) ){
		T xi = in[idx];
		T result=T(0);

		vector_td<int,D> co = idx_to_co(idx, dims);

                REAL grad = dpct::max(gradient<REAL, T, D>(in, dims, co), limit);

                if (grad > limit) {
			//result += REAL(D)*xi/grad;
			for (int i = 0; i < D; i++){
				co[i]+=1;
				result += (xi-in[co_to_idx((co+dims)%dims,dims)])/grad;
				co[i]-=1;
			}
		}

		for (int i = 0; i < D; i++){
			co[i]-=1;
                        grad = dpct::max(gradient<REAL, T, D>(in, dims, co), limit);

                        if (grad > limit) {
				result +=(xi-in[co_to_idx((co+dims)%dims,dims)])/grad;
			}
			co[i]+=1;
		}
		out[idx] += result*weight;
	}
}








template<class T, unsigned int D> void cuTvOperator<T,D>::gradient (cuNDArray<T> * in,cuNDArray<T> * out, bool accumulate)
{
	if (!accumulate)
		clear(out);

	const typename intd<D>::Type dims = vector_td<int,D>( from_std_vector<size_t,D>(in->get_dimensions()));
	int elements = in->get_number_of_elements();

	int threadsPerBlock =std::min(prod(dims),256); //Using hardcoded blockSize because we use quite a lot of registers

        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGridx = std::min(std::max(1,prod(dims)/threadsPerBlock),cudaDeviceManager::Instance()->max_griddim());
	int totalBlocksPerGridy = (prod(dims)-1)/(threadsPerBlock*totalBlocksPerGridx)+1;
        dpct::dim3 dimGrid(totalBlocksPerGridx, totalBlocksPerGridy);

        for (int i =0; i < (elements/prod(dims)); i++){
                /*
                DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit,
                query info::device::max_work_group_size. Adjust the work-group size if needed.
                */
                dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

                dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                        auto in_get_data_ptr_i_prod_dims_ct0 = in->get_data_ptr() + i * prod(dims);
                        auto out_get_data_ptr_i_prod_dims_ct1 = out->get_data_ptr() + i * prod(dims);
                        auto limit__ct3 = limit_;
                        auto this_weight__ct4 = this->weight_;

                        cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

                        cgh.parallel_for<dpct_kernel_name<class tvGradient_kernel_16856b, REAL,
                                                          T,
                                                          dpct_kernel_scalar<D>>>(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                                    tvGradient_kernel(in_get_data_ptr_i_prod_dims_ct0, out_get_data_ptr_i_prod_dims_ct1,
                                                      dims, limit__ct3, this_weight__ct4);
                            });
                });
        }

	//cudaDeviceSynchronize();
	//CHECK_FOR_CUDA_ERROR();
}

template<class REAL, class T, unsigned int D> static void tvMagnitude_kernel(const  T* __restrict__  in,T* __restrict__ out,const vector_td<int,D> dims,REAL limit,REAL weight)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if( idx < prod(dims) ){
		vector_td<int,D> co = idx_to_co(idx, dims);
		REAL grad = gradient<REAL,T,D>(in,dims,co);
		//out[idx] =  (grad > limit) ? grad*weight : REAL(0);
		out[idx] = grad*weight;
	}
}


template<class T, unsigned int D> typename realType<T>::Type cuTvOperator<T,D>::magnitude (cuNDArray<T> * in)
{
	cuNDArray<T> out(in->get_dimensions());
	const typename intd<D>::Type dims = vector_td<int,D>( from_std_vector<size_t,D>(in->get_dimensions()));
	int elements = in->get_number_of_elements();

	int threadsPerBlock =std::min(prod(dims),256); //Using hardcoded blockSize because we use quite a lot of registers
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGridx = std::min(std::max(1,prod(dims)/threadsPerBlock),cudaDeviceManager::Instance()->max_griddim());
	int totalBlocksPerGridy = (prod(dims)-1)/(threadsPerBlock*totalBlocksPerGridx)+1;
        dpct::dim3 dimGrid(totalBlocksPerGridx, totalBlocksPerGridy);

        for (int i =0; i < (elements/prod(dims)); i++){
                /*
                DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit,
                query info::device::max_work_group_size. Adjust the work-group size if needed.
                */
                dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

                dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                        auto in_get_data_ptr_i_prod_dims_ct0 = in->get_data_ptr() + i * prod(dims);
                        auto out_get_data_ptr_i_prod_dims_ct1 = out.get_data_ptr() + i * prod(dims);
                        auto limit__ct3 = limit_;
                        auto this_weight__ct4 = this->weight_;

                        cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

                        cgh.parallel_for<dpct_kernel_name<class tvMagnitude_kernel_862878, REAL,
                                                          T,
                                                          dpct_kernel_scalar<D>>>(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                                    tvMagnitude_kernel(in_get_data_ptr_i_prod_dims_ct0,
                                                       out_get_data_ptr_i_prod_dims_ct1, dims, limit__ct3,
                                                       this_weight__ct4);
                            });
                });
        }

	//cudaDeviceSynchronize();
	//CHECK_FOR_CUDA_ERROR();
	return asum(&out);
}

namespace Gadgetron {
template class EXPORTGPUOPERATORS cuTvOperator<float,1>;
template class EXPORTGPUOPERATORS cuTvOperator<float,2>;
template class EXPORTGPUOPERATORS cuTvOperator<float,3>;
template class EXPORTGPUOPERATORS cuTvOperator<float,4>;

template class EXPORTGPUOPERATORS cuTvOperator<double,1>;
template class EXPORTGPUOPERATORS cuTvOperator<double,2>;
template class EXPORTGPUOPERATORS cuTvOperator<double,3>;
template class EXPORTGPUOPERATORS cuTvOperator<double,4>;

template class EXPORTGPUOPERATORS cuTvOperator<float_complext,1>;
template class EXPORTGPUOPERATORS cuTvOperator<float_complext,2>;
template class EXPORTGPUOPERATORS cuTvOperator<float_complext,3>;
template class EXPORTGPUOPERATORS cuTvOperator<float_complext,4>;

template class EXPORTGPUOPERATORS cuTvOperator<double_complext,1>;
template class EXPORTGPUOPERATORS cuTvOperator<double_complext,2>;
template class EXPORTGPUOPERATORS cuTvOperator<double_complext,3>;
template class EXPORTGPUOPERATORS cuTvOperator<double_complext,4>;
}
