#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "complext.h"
#include "cuSolverUtils.h"
#include <dpct/dpl_utils.hpp>

/* DPCT_ORIG #include <thrust/transform.h>*/
/* DPCT_ORIG #include <thrust/iterator/zip_iterator.h>*/
#include "cuNDArray_math.h"
#include <cmath>

#define MAX_THREADS_PER_BLOCK 512

using namespace Gadgetron;
/* DPCT_ORIG template <class T> __global__ static void filter_kernel(T* x, T* g, int elements){*/
template <class T> static void filter_kernel(T* x, T* g, int elements) {
/* DPCT_ORIG 	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < elements){
		if ( x[idx] <= T(0) && g[idx] > 0) g[idx]=T(0);
	}
}

/* DPCT_ORIG template <class REAL> __global__ static void filter_kernel(complext<REAL>* x, complext<REAL>* g, int
 * elements){*/
template <class REAL> static void filter_kernel(complext<REAL>* x, complext<REAL>* g, int elements) {
/* DPCT_ORIG 	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < elements){
		if ( real(x[idx]) <= REAL(0) && real(g[idx]) > 0) g[idx]._real = REAL(0);
		g[idx]._imag=REAL(0);
	}
}

template <class T> void EXPORTGPUSOLVERS Gadgetron::solver_non_negativity_filter(cuNDArray<T>* x , cuNDArray<T>* g)
{
	int elements = g->get_number_of_elements();

	int threadsPerBlock = std::min(elements,MAX_THREADS_PER_BLOCK);
/* DPCT_ORIG 	dim3 dimBlock( threadsPerBlock);*/
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = std::max(1,elements/MAX_THREADS_PER_BLOCK);
/* DPCT_ORIG 	dim3 dimGrid(totalBlocksPerGrid);*/
        dpct::dim3 dimGrid(totalBlocksPerGrid);

/* DPCT_ORIG 	filter_kernel<typename
 * realType<T>::Type><<<dimGrid,dimBlock>>>(x->get_data_ptr(),g->get_data_ptr(),elements);*/
        /*
        DPCT1049:116: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto x_get_data_ptr_ct0 = x->get_data_ptr();
            auto g_get_data_ptr_ct1 = g->get_data_ptr();

            cgh.parallel_for<dpct_kernel_name<class filter_kernel_2c7f8a, typename realType<T>::Type>>(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                    filter_kernel<typename realType<T>::Type>(x_get_data_ptr_ct0, g_get_data_ptr_ct1, elements);
                });
        });
    }
}



template<class T> struct updateF_functor{

	typedef typename realType<T>::Type REAL;
	updateF_functor(REAL alpha_, REAL sigma_){

		alpha= alpha_;
		sigma = sigma_;
	}
/* DPCT_ORIG 	__device__ __inline__ T operator() (T val){*/
        __inline__ T operator()(T val) const {
/* DPCT_ORIG 		return val/(1+alpha*sigma)/max(REAL(1),abs(val/(1+alpha*sigma)));*/
                /*
                DPCT1064:255: Migrated abs call is used in a macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                /*
                DPCT1064:256: Migrated max call is used in a macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                return val / (1 + alpha * sigma) / dpct::max(REAL(1), sycl::fabs(val / (1 + alpha * sigma)));
        }
	typename realType<T>::Type alpha, sigma;
};

template<class T>
inline void Gadgetron::updateF(cuNDArray<T>& data,
		typename realType<T>::Type alpha, typename realType<T>::Type sigma) {
/* DPCT_ORIG 	thrust::transform(data.begin(),data.end(),data.begin(),updateF_functor<T>(alpha,sigma));*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data.begin(), data.end(),
                       data.begin(), updateF_functor<T>(alpha, sigma));
}


template<class T> struct updateFgroup_functor {

	typedef typename realType<T>::Type REAL;
	updateFgroup_functor(REAL alpha_, REAL sigma_) : alpha(alpha_), sigma(sigma_){
	}

/* DPCT_ORIG 	__device__ __inline__ T operator() (thrust::tuple<T,typename realType<T>::Type> tup){*/
        __inline__ T operator()(std::tuple<T, typename realType<T>::Type> tup) const {
/* DPCT_ORIG 			return
 * thrust::get<0>(tup)/(1+alpha*sigma)/max(REAL(1),thrust::get<1>(tup)/(1+alpha*sigma));*/
                        return std::get<0>(tup) / (1 + alpha * sigma) / dpct::max(REAL(1), std::get<1>(tup) / (1 + alpha * sigma));
        }

	typename realType<T>::Type alpha, sigma;
};

template<class T> struct add_square_functor{

/* DPCT_ORIG 	__device__ __inline__ typename realType<T>::Type operator() (thrust::tuple<T,typename realType<T>::Type>
 * tup){*/
        __inline__ typename realType<T>::Type operator()(std::tuple<T, typename realType<T>::Type> tup) const {
/* DPCT_ORIG 		T val = thrust::get<0>(tup);*/
                T val = std::get<0>(tup);
/* DPCT_ORIG 		return thrust::get<1>(tup)+norm(val);*/
                /*
                DPCT1017:257: The dpct::length call is used instead of the norm call. These two calls do not provide
                exactly the same functionality. Check the potential precision and/or performance issues for the
                generated code.
                */
                return std::get<1>(tup) + norm(val);
        }
};
template<class T>
inline void Gadgetron::updateFgroup(std::vector<cuNDArray<T> >& datas,
		typename realType<T>::Type alpha, typename realType<T>::Type sigma) {

	cuNDArray<typename realType<T>::Type> squares(datas.front().get_dimensions());
	clear(&squares);
	for (int i = 0; i < datas.size(); i++)
/* DPCT_ORIG thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(datas[i].begin(),squares.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(datas[i].end(),squares.end())),
   squares.begin(), add_square_functor<T>());*/
                std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()),
                               oneapi::dpl::make_zip_iterator(std::make_tuple(datas[i].begin(), squares.begin())),
                               oneapi::dpl::make_zip_iterator(std::make_tuple(datas[i].end(), squares.end())),
                               squares.begin(), add_square_functor<T>());

        sqrt_inplace(&squares);
	for (int i = 0 ; i < datas.size(); i++){
/* DPCT_ORIG thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(datas[i].begin(),squares.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(datas[i].end(),squares.end())),
   datas[i].begin(), updateFgroup_functor<T>(alpha,sigma));*/
                std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()),
                               oneapi::dpl::make_zip_iterator(std::make_tuple(datas[i].begin(), squares.begin())),
                               oneapi::dpl::make_zip_iterator(std::make_tuple(datas[i].end(), squares.end())),
                               datas[i].begin(), updateFgroup_functor<T>(alpha, sigma));
        }
}

template void EXPORTGPUSOLVERS Gadgetron::updateF<float>(cuNDArray<float>& data, float alpha, float sigma);
template void EXPORTGPUSOLVERS Gadgetron::updateF<double>(cuNDArray<double>& data,double alpha, double sigma);
template void EXPORTGPUSOLVERS Gadgetron::updateF<float_complext>(cuNDArray<float_complext>& data, float alpha, float sigma);
template void EXPORTGPUSOLVERS Gadgetron::updateF<double_complext>(cuNDArray<double_complext>& data, double alpha, double sigma);

template void EXPORTGPUSOLVERS Gadgetron::updateFgroup<float>(std::vector<cuNDArray<float> >& data, float alpha, float sigma);
template void EXPORTGPUSOLVERS Gadgetron::updateFgroup<double>(std::vector<cuNDArray<double> >& data,double alpha, double sigma);
template void EXPORTGPUSOLVERS Gadgetron::updateFgroup<float_complext>(std::vector<cuNDArray<float_complext> >& data, float alpha, float sigma);
template void EXPORTGPUSOLVERS Gadgetron::updateFgroup<double_complext>(std::vector<cuNDArray<double_complext> >& data, double alpha, double sigma);


template void EXPORTGPUSOLVERS Gadgetron::solver_non_negativity_filter<float>(cuNDArray<float>*, cuNDArray<float>*);
template void EXPORTGPUSOLVERS Gadgetron::solver_non_negativity_filter<double>(cuNDArray<double>*, cuNDArray<double>*);
template void EXPORTGPUSOLVERS Gadgetron::solver_non_negativity_filter<float_complext>(cuNDArray<float_complext>*, cuNDArray<float_complext>*);
template void EXPORTGPUSOLVERS Gadgetron::solver_non_negativity_filter<double_complext>(cuNDArray<double_complext>*, cuNDArray<double_complext>*);

