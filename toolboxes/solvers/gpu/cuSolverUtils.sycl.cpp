#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "complext.h"
#include "cuSolverUtils.h"
#include <dpct/dpl_utils.hpp>

#include "cuNDArray_math.h"
#include <cmath>

// #include <sycl/ext/intel/math.hpp>
#include <oneapi/math.hpp>

#define MAX_THREADS_PER_BLOCK 512

using namespace Gadgetron;
template <class T> static void filter_kernel(T* x, T* g, int elements){
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < elements){
		if ( x[idx] <= T(0) && g[idx] > 0) g[idx]=T(0);
	}
}

template <class REAL> static void filter_kernel(complext<REAL>* x, complext<REAL>* g, int elements){
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
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = std::max(1,elements/MAX_THREADS_PER_BLOCK);
        dpct::dim3 dimGrid(totalBlocksPerGrid);

        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
                auto exp_props =
                    sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

                dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                        auto x_get_data_ptr_ct0 = x->get_data_ptr();
                        auto g_get_data_ptr_ct1 = g->get_data_ptr();

                        cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

                        cgh.parallel_for<dpct_kernel_name<class filter_kernel_2c7f8a, T>>(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), exp_props, [=](sycl::nd_item<3> item_ct1) {
                                    filter_kernel<typename realType<T>::Type>(x_get_data_ptr_ct0, g_get_data_ptr_ct1,
                                                                              elements);
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
        __inline__ T operator()(T val) const {
                /*
                DPCT1064:32: Migrated max call is used in a macro/template definition and may not be valid for all
                macro/template uses. Adjust the code.
                */
                return val / (1 + alpha * sigma) / dpct::max(REAL(1), abs(val / (1 + alpha * sigma)));
        }
	typename realType<T>::Type alpha, sigma;
};

template<class T>
inline void Gadgetron::updateF(cuNDArray<T>& data,
		typename realType<T>::Type alpha, typename realType<T>::Type sigma) {
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data.begin(), data.end(),
                       data.begin(), updateF_functor<T>(alpha, sigma));
}


template<class T> struct updateFgroup_functor {

	typedef typename realType<T>::Type REAL;
	updateFgroup_functor(REAL alpha_, REAL sigma_) : alpha(alpha_), sigma(sigma_){
	}

        __inline__ T operator()(std::tuple<T, typename realType<T>::Type> tup) const {
                        return std::get<0>(tup) / (1 + alpha * sigma) / dpct::max(REAL(1), std::get<1>(tup) / (1 + alpha * sigma));
        }

	typename realType<T>::Type alpha, sigma;
};

template<class T> struct add_square_functor{

        __inline__ typename realType<T>::Type operator()(std::tuple<T, typename realType<T>::Type> tup) const {
                T val = std::get<0>(tup);
                return std::get<1>(tup) + norm(val);
        }
};
template<class T>
inline void Gadgetron::updateFgroup(std::vector<cuNDArray<T> >& datas,
		typename realType<T>::Type alpha, typename realType<T>::Type sigma) {

	cuNDArray<typename realType<T>::Type> squares(datas.front().get_dimensions());
	clear(&squares);
	for (int i = 0; i < datas.size(); i++)
                std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()),
                               oneapi::dpl::make_zip_iterator(std::make_tuple(datas[i].begin(), squares.begin())),
                               oneapi::dpl::make_zip_iterator(std::make_tuple(datas[i].end(), squares.end())),
                               squares.begin(), add_square_functor<T>());

        sqrt_inplace(&squares);
	for (int i = 0 ; i < datas.size(); i++){
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

