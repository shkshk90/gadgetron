#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuVector_td_test_kernels.h"
#include "check_CUDA.h"
#include "vector_td_utilities.h"
#include "cuNDArray.h"
#include "cudaDeviceManager.h"
#include <dpct/dpl_utils.hpp>

#include <cmath>

/* DPCT_ORIG #include "thrust/device_vector.h"*/

using namespace Gadgetron;
/* DPCT_ORIG template<class T, unsigned int D> __global__ void abs_kernel(vector_td<T,D>* data, unsigned int size){*/
template <class T, unsigned int D> void abs_kernel(vector_td<T, D>* data, unsigned int size) {
/* DPCT_ORIG 	 const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
         auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
         const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
         if (idx < size) data[idx] = abs(data[idx]);
}


template<class T, unsigned int D> void Gadgetron::test_abs(cuNDArray< vector_td<T,D> >* data){

/* DPCT_ORIG 	dim3
 * dimBlock(std::min(cudaDeviceManager::Instance()->max_griddim(),(int)data->get_number_of_elements()));*/
        dpct::dim3 dimBlock(std::min(cudaDeviceManager::Instance()->max_griddim(), (int)data->get_number_of_elements()));
/* DPCT_ORIG 	dim3 dimGrid((dimBlock.x-1)/data->get_number_of_elements()+1);*/
        dpct::dim3 dimGrid((dimBlock.x - 1) / data->get_number_of_elements() + 1);
/* DPCT_ORIG 	abs_kernel<<<dimGrid,dimBlock>>>(data->get_data_ptr(),data->get_number_of_elements());*/
        /*
        DPCT1049:117: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto data_get_data_ptr_ct0 = data->get_data_ptr();
            auto data_get_number_of_elements_ct1 = data->get_number_of_elements();

            /*
            DPCT1050:299: The template argument of the dpct_kernel_name could not be deduced. You need to update this
            code.
            */
            cgh.parallel_for(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                    abs_kernel(data_get_data_ptr_ct0, data_get_number_of_elements_ct1);
                });
        });
    }
/* DPCT_ORIG 	cudaDeviceSynchronize();*/
        dpct::get_current_device().queues_wait_and_throw();
        CHECK_FOR_CUDA_ERROR();
}

template <typename T, unsigned int D>
/* DPCT_ORIG struct test_norm_functor : public thrust::unary_function<T,vector_td<T,D> >*/
/*
DPCT1044:258: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need
to remove references to typedefs from thrust::unary_function in the class definition.
*/
struct test_norm_functor {
/* DPCT_ORIG  __host__ __device__ T operator()(const vector_td<T,D> &x) const {return norm(x);}*/
 T operator()(const vector_td<T, D>& x) const { return norm(x); }
};
/* DPCT_ORIG template<class T, unsigned int D> thrust::device_vector<T> Gadgetron::test_norm(cuNDArray< vector_td<T,D>
 * >* data){*/
template <class T, unsigned int D> dpct::device_vector<T> Gadgetron::test_norm(cuNDArray<vector_td<T, D>>* data) {

/* DPCT_ORIG 	thrust::device_vector<T> out(data->get_number_of_elements());*/
        dpct::device_vector<T> out(data->get_number_of_elements());
/* DPCT_ORIG 	thrust::transform(data->begin(),data->end(),out.begin(),test_norm_functor<T,D>());*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data->begin(),
                       data->end(), out.begin(), test_norm_functor<T, D>());
/* DPCT_ORIG 	cudaDeviceSynchronize();*/
        dpct::get_current_device().queues_wait_and_throw();
        CHECK_FOR_CUDA_ERROR();
	return out;
}

template <typename T, unsigned int D>
/* DPCT_ORIG struct test_min_functor : public thrust::unary_function<T,vector_td<T,D> >*/
/*
DPCT1044:259: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need
to remove references to typedefs from thrust::unary_function in the class definition.
*/
struct test_min_functor {
/* DPCT_ORIG  __host__ __device__ T operator()(const vector_td<T,D> &x) const {return min(x);}*/
 T operator()(const vector_td<T, D>& x) const { return min(x); }
};
/* DPCT_ORIG template<class T, unsigned int D> thrust::device_vector<T> Gadgetron::test_min(cuNDArray< vector_td<T,D> >*
 * data){*/
template <class T, unsigned int D> dpct::device_vector<T> Gadgetron::test_min(cuNDArray<vector_td<T, D>>* data) {

/* DPCT_ORIG 	thrust::device_vector<T> out(data->get_number_of_elements());*/
        dpct::device_vector<T> out(data->get_number_of_elements());
/* DPCT_ORIG 	thrust::transform(data->begin(),data->end(),out.begin(),test_min_functor<T,D>());*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data->begin(),
                       data->end(), out.begin(), test_min_functor<T, D>());
/* DPCT_ORIG 	cudaDeviceSynchronize();*/
        dpct::get_current_device().queues_wait_and_throw();
        CHECK_FOR_CUDA_ERROR();
	return out;
}

template <typename T, unsigned int D>
/* DPCT_ORIG struct test_max_functor : public thrust::unary_function<T,vector_td<T,D> >*/
/*
DPCT1044:260: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need
to remove references to typedefs from thrust::unary_function in the class definition.
*/
struct test_max_functor {
/* DPCT_ORIG  __host__ __device__ T operator()(const vector_td<T,D> &x) const {return max(x);}*/
 T operator()(const vector_td<T, D>& x) const { return max(x); }
};
/* DPCT_ORIG template<class T, unsigned int D> thrust::device_vector<T> Gadgetron::test_max(cuNDArray< vector_td<T,D> >*
 * data){*/
template <class T, unsigned int D> dpct::device_vector<T> Gadgetron::test_max(cuNDArray<vector_td<T, D>>* data) {

/* DPCT_ORIG 	thrust::device_vector<T> out(data->get_number_of_elements());*/
        dpct::device_vector<T> out(data->get_number_of_elements());
/* DPCT_ORIG 	thrust::transform(data->begin(),data->end(),out.begin(),test_max_functor<T,D>());*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data->begin(),
                       data->end(), out.begin(), test_max_functor<T, D>());
/* DPCT_ORIG 	cudaDeviceSynchronize();*/
        dpct::get_current_device().queues_wait_and_throw();
        CHECK_FOR_CUDA_ERROR();
	return out;
}

template <typename T, unsigned int D>
/* DPCT_ORIG struct test_amin_functor : public thrust::binary_function<vector_td<T,D>, vector_td<T,D>, vector_td<T,D>
   >*/
/*
DPCT1044:261: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
need to remove references to typedefs from thrust::binary_function in the class definition.
*/
struct test_amin_functor {
/* DPCT_ORIG 	__host__ __device__ vector_td<T,D> operator()(const vector_td<T,D> &x, const vector_td<T,D> &y) const
 * {return amin(x,y);}*/
        vector_td<T, D> operator()(const vector_td<T, D>& x, const vector_td<T, D>& y) const { return amin(x, y); }
};

template<class T, unsigned int D> boost::shared_ptr<cuNDArray<vector_td<T,D> > > Gadgetron::test_amin(cuNDArray< vector_td<T,D> >* data1, cuNDArray< vector_td<T,D> >* data2){
	boost::shared_ptr<cuNDArray<vector_td<T,D> > > out( new cuNDArray<vector_td<T,D> >(data1->get_dimensions()));
/* DPCT_ORIG 	thrust::transform(data1->begin(),data1->end(),data2->begin(),out->begin(),test_amin_functor<T,D>());*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data1->begin(),
                       data1->end(), data2->begin(), out->begin(), test_amin_functor<T, D>());
        return out;
}

template <typename T, unsigned int D>
/* DPCT_ORIG struct test_amax_functor : public thrust::binary_function<vector_td<T,D>, vector_td<T,D>, vector_td<T,D>
   >*/
/*
DPCT1044:262: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
need to remove references to typedefs from thrust::binary_function in the class definition.
*/
struct test_amax_functor {
/* DPCT_ORIG 	__host__ __device__ vector_td<T,D> operator()(const vector_td<T,D> &x, const vector_td<T,D> &y) const
 * {return amax(x,y);}*/
        vector_td<T, D> operator()(const vector_td<T, D>& x, const vector_td<T, D>& y) const { return amax(x, y); }
};

template<class T, unsigned int D> boost::shared_ptr<cuNDArray<vector_td<T,D> > > Gadgetron::test_amax(cuNDArray< vector_td<T,D> >* data1, cuNDArray< vector_td<T,D> >* data2){
	boost::shared_ptr<cuNDArray<vector_td<T,D> > > out( new cuNDArray<vector_td<T,D> >(data1->get_dimensions()));
/* DPCT_ORIG 	thrust::transform(data1->begin(),data1->end(),data2->begin(),out->begin(),test_amax_functor<T,D>());*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data1->begin(),
                       data1->end(), data2->begin(), out->begin(), test_amax_functor<T, D>());
        return out;
}

template <typename T, unsigned int D>
/* DPCT_ORIG class test_amin2_functor : public thrust::unary_function<vector_td<T,D>, vector_td<T,D> >*/
/*
DPCT1044:263: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need
to remove references to typedefs from thrust::unary_function in the class definition.
*/
class test_amin2_functor {
public:
	test_amin2_functor(T _val): val(_val){};
/* DPCT_ORIG 	__host__ __device__ vector_td<T,D> operator()(const vector_td<T,D> &x) const {return amin(x,val);}*/
        vector_td<T, D> operator()(const vector_td<T, D>& x) const { return amin(x, val); }
        T val;
};

template<class T, unsigned int D> boost::shared_ptr<cuNDArray<vector_td<T,D> > > Gadgetron::test_amin2(cuNDArray< vector_td<T,D> >* data1, T val){
	boost::shared_ptr<cuNDArray<vector_td<T,D> > > out( new cuNDArray<vector_td<T,D> >(data1->get_dimensions()));
/* DPCT_ORIG 	thrust::transform(data1->begin(),data1->end(),out->begin(),test_amin2_functor<T,D>(val));*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data1->begin(),
                       data1->end(), out->begin(), test_amin2_functor<T, D>(val));
        return out;
}

template <typename T, unsigned int D>
/* DPCT_ORIG class test_amax2_functor : public thrust::unary_function<vector_td<T,D>, vector_td<T,D> >*/
/*
DPCT1044:264: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need
to remove references to typedefs from thrust::unary_function in the class definition.
*/
class test_amax2_functor {
public:
	test_amax2_functor(T _val): val(_val){};
/* DPCT_ORIG 	__host__ __device__ vector_td<T,D> operator()(const vector_td<T,D> &x) const {return amax(x,val);}*/
        vector_td<T, D> operator()(const vector_td<T, D>& x) const { return amax(x, val); }
        T val;
};

template<class T, unsigned int D> boost::shared_ptr<cuNDArray<vector_td<T,D> > > Gadgetron::test_amax2(cuNDArray< vector_td<T,D> >* data1, T val){
	boost::shared_ptr<cuNDArray<vector_td<T,D> > > out( new cuNDArray<vector_td<T,D> >(data1->get_dimensions()));
/* DPCT_ORIG 	thrust::transform(data1->begin(),data1->end(),out->begin(),test_amax2_functor<T,D>(val));*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data1->begin(),
                       data1->end(), out->begin(), test_amax2_functor<T, D>(val));
        return out;
}



template<class T, unsigned int D> void Gadgetron::vector_fill(cuNDArray< vector_td<T,D> >* data,  vector_td<T,D> val){
/* DPCT_ORIG 	thrust::fill(data->begin(),data->end(),val);*/
        std::fill(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), data->begin(), data->end(), val);
}


template void Gadgetron::test_abs<float,1>(cuNDArray< vector_td<float,1> > *);
template void Gadgetron::test_abs<float,2>(cuNDArray< vector_td<float,2> > *);
template  void Gadgetron::test_abs<float,3>(cuNDArray< vector_td<float,3> > *);
template  void Gadgetron::test_abs<float,4>(cuNDArray< vector_td<float,4> > *);

template  void Gadgetron::test_abs<double,1>(cuNDArray< vector_td<double,1> > *);
template void Gadgetron::test_abs<double,2>(cuNDArray< vector_td<double,2> > *);
template void Gadgetron::test_abs<double,3>(cuNDArray< vector_td<double,3> > *);
template void Gadgetron::test_abs<double,4>(cuNDArray< vector_td<double,4> > *);

/* DPCT_ORIG template thrust::device_vector<float> Gadgetron::test_norm<float,1>(cuNDArray< vector_td<float,1> > *);*/
template dpct::device_vector<float> Gadgetron::test_norm<float, 1>(cuNDArray<vector_td<float, 1>>*);
/* DPCT_ORIG template thrust::device_vector<float> Gadgetron::test_norm<float,2>(cuNDArray< vector_td<float,2> > *);*/
template dpct::device_vector<float> Gadgetron::test_norm<float, 2>(cuNDArray<vector_td<float, 2>>*);
/* DPCT_ORIG template  thrust::device_vector<float> Gadgetron::test_norm<float,3>(cuNDArray< vector_td<float,3> > *);*/
template dpct::device_vector<float> Gadgetron::test_norm<float, 3>(cuNDArray<vector_td<float, 3>>*);
/* DPCT_ORIG template  thrust::device_vector<float> Gadgetron::test_norm<float,4>(cuNDArray< vector_td<float,4> > *);*/
template dpct::device_vector<float> Gadgetron::test_norm<float, 4>(cuNDArray<vector_td<float, 4>>*);

/* DPCT_ORIG template  thrust::device_vector<double> Gadgetron::test_norm<double,1>(cuNDArray< vector_td<double,1> >
 * *);*/
template dpct::device_vector<double> Gadgetron::test_norm<double, 1>(cuNDArray<vector_td<double, 1>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_norm<double,2>(cuNDArray< vector_td<double,2> >
 * *);*/
template dpct::device_vector<double> Gadgetron::test_norm<double, 2>(cuNDArray<vector_td<double, 2>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_norm<double,3>(cuNDArray< vector_td<double,3> >
 * *);*/
template dpct::device_vector<double> Gadgetron::test_norm<double, 3>(cuNDArray<vector_td<double, 3>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_norm<double,4>(cuNDArray< vector_td<double,4> >
 * *);*/
template dpct::device_vector<double> Gadgetron::test_norm<double, 4>(cuNDArray<vector_td<double, 4>>*);

/* DPCT_ORIG template thrust::device_vector<float> Gadgetron::test_min<float,1>(cuNDArray< vector_td<float,1> > *);*/
template dpct::device_vector<float> Gadgetron::test_min<float, 1>(cuNDArray<vector_td<float, 1>>*);
/* DPCT_ORIG template thrust::device_vector<float> Gadgetron::test_min<float,2>(cuNDArray< vector_td<float,2> > *);*/
template dpct::device_vector<float> Gadgetron::test_min<float, 2>(cuNDArray<vector_td<float, 2>>*);
/* DPCT_ORIG template  thrust::device_vector<float> Gadgetron::test_min<float,3>(cuNDArray< vector_td<float,3> > *);*/
template dpct::device_vector<float> Gadgetron::test_min<float, 3>(cuNDArray<vector_td<float, 3>>*);
/* DPCT_ORIG template  thrust::device_vector<float> Gadgetron::test_min<float,4>(cuNDArray< vector_td<float,4> > *);*/
template dpct::device_vector<float> Gadgetron::test_min<float, 4>(cuNDArray<vector_td<float, 4>>*);

/* DPCT_ORIG template  thrust::device_vector<double> Gadgetron::test_min<double,1>(cuNDArray< vector_td<double,1> >
 * *);*/
template dpct::device_vector<double> Gadgetron::test_min<double, 1>(cuNDArray<vector_td<double, 1>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_min<double,2>(cuNDArray< vector_td<double,2> > *);*/
template dpct::device_vector<double> Gadgetron::test_min<double, 2>(cuNDArray<vector_td<double, 2>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_min<double,3>(cuNDArray< vector_td<double,3> > *);*/
template dpct::device_vector<double> Gadgetron::test_min<double, 3>(cuNDArray<vector_td<double, 3>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_min<double,4>(cuNDArray< vector_td<double,4> > *);*/
template dpct::device_vector<double> Gadgetron::test_min<double, 4>(cuNDArray<vector_td<double, 4>>*);

/* DPCT_ORIG template thrust::device_vector<float> Gadgetron::test_max<float,1>(cuNDArray< vector_td<float,1> > *);*/
template dpct::device_vector<float> Gadgetron::test_max<float, 1>(cuNDArray<vector_td<float, 1>>*);
/* DPCT_ORIG template thrust::device_vector<float> Gadgetron::test_max<float,2>(cuNDArray< vector_td<float,2> > *);*/
template dpct::device_vector<float> Gadgetron::test_max<float, 2>(cuNDArray<vector_td<float, 2>>*);
/* DPCT_ORIG template  thrust::device_vector<float> Gadgetron::test_max<float,3>(cuNDArray< vector_td<float,3> > *);*/
template dpct::device_vector<float> Gadgetron::test_max<float, 3>(cuNDArray<vector_td<float, 3>>*);
/* DPCT_ORIG template  thrust::device_vector<float> Gadgetron::test_max<float,4>(cuNDArray< vector_td<float,4> > *);*/
template dpct::device_vector<float> Gadgetron::test_max<float, 4>(cuNDArray<vector_td<float, 4>>*);

/* DPCT_ORIG template  thrust::device_vector<double> Gadgetron::test_max<double,1>(cuNDArray< vector_td<double,1> >
 * *);*/
template dpct::device_vector<double> Gadgetron::test_max<double, 1>(cuNDArray<vector_td<double, 1>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_max<double,2>(cuNDArray< vector_td<double,2> > *);*/
template dpct::device_vector<double> Gadgetron::test_max<double, 2>(cuNDArray<vector_td<double, 2>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_max<double,3>(cuNDArray< vector_td<double,3> > *);*/
template dpct::device_vector<double> Gadgetron::test_max<double, 3>(cuNDArray<vector_td<double, 3>>*);
/* DPCT_ORIG template thrust::device_vector<double> Gadgetron::test_max<double,4>(cuNDArray< vector_td<double,4> > *);*/
template dpct::device_vector<double> Gadgetron::test_max<double, 4>(cuNDArray<vector_td<double, 4>>*);

template boost::shared_ptr<cuNDArray<vector_td<float,1> > > Gadgetron::test_amin<float,1>(cuNDArray< vector_td<float,1> > *,cuNDArray< vector_td<float,1> > *);
template boost::shared_ptr<cuNDArray<vector_td<float,2> > > Gadgetron::test_amin<float,2>(cuNDArray< vector_td<float,2> > *, cuNDArray< vector_td<float,2> > *);
template  boost::shared_ptr<cuNDArray<vector_td<float,3> > > Gadgetron::test_amin<float,3>(cuNDArray< vector_td<float,3> > *, cuNDArray< vector_td<float,3> > *);
template  boost::shared_ptr<cuNDArray<vector_td<float,4> > > Gadgetron::test_amin<float,4>(cuNDArray< vector_td<float,4> > *, cuNDArray< vector_td<float,4> > *);

template  boost::shared_ptr<cuNDArray<vector_td<double,1> > > Gadgetron::test_amin<double,1>(cuNDArray< vector_td<double,1> > *, cuNDArray< vector_td<double,1> > *);
template boost::shared_ptr<cuNDArray<vector_td<double,2> > > Gadgetron::test_amin<double,2>(cuNDArray< vector_td<double,2> > *, cuNDArray< vector_td<double,2> > *);
template boost::shared_ptr<cuNDArray<vector_td<double,3> > > Gadgetron::test_amin<double,3>(cuNDArray< vector_td<double,3> > *, cuNDArray< vector_td<double,3> > *);
template boost::shared_ptr<cuNDArray<vector_td<double,4> > > Gadgetron::test_amin<double,4>(cuNDArray< vector_td<double,4> > *, cuNDArray< vector_td<double,4> > *);



template boost::shared_ptr<cuNDArray<vector_td<float,1> > > Gadgetron::test_amin2<float,1>(cuNDArray< vector_td<float,1> > *, float );
template boost::shared_ptr<cuNDArray<vector_td<float,2> > > Gadgetron::test_amin2<float,2>(cuNDArray< vector_td<float,2> > *, float);
template  boost::shared_ptr<cuNDArray<vector_td<float,3> > > Gadgetron::test_amin2<float,3>(cuNDArray< vector_td<float,3> > *, float);
template  boost::shared_ptr<cuNDArray<vector_td<float,4> > > Gadgetron::test_amin2<float,4>(cuNDArray< vector_td<float,4> > *, float);

template  boost::shared_ptr<cuNDArray<vector_td<double,1> > > Gadgetron::test_amin2<double,1>(cuNDArray< vector_td<double,1> > *, double);
template boost::shared_ptr<cuNDArray<vector_td<double,2> > > Gadgetron::test_amin2<double,2>(cuNDArray< vector_td<double,2> > *, double);
template boost::shared_ptr<cuNDArray<vector_td<double,3> > > Gadgetron::test_amin2<double,3>(cuNDArray< vector_td<double,3> > *, double);
template boost::shared_ptr<cuNDArray<vector_td<double,4> > > Gadgetron::test_amin2<double,4>(cuNDArray< vector_td<double,4> > *, double);



template boost::shared_ptr<cuNDArray<vector_td<float,1> > > Gadgetron::test_amax<float,1>(cuNDArray< vector_td<float,1> > *,cuNDArray< vector_td<float,1> > *);
template boost::shared_ptr<cuNDArray<vector_td<float,2> > > Gadgetron::test_amax<float,2>(cuNDArray< vector_td<float,2> > *, cuNDArray< vector_td<float,2> > *);
template  boost::shared_ptr<cuNDArray<vector_td<float,3> > > Gadgetron::test_amax<float,3>(cuNDArray< vector_td<float,3> > *, cuNDArray< vector_td<float,3> > *);
template  boost::shared_ptr<cuNDArray<vector_td<float,4> > > Gadgetron::test_amax<float,4>(cuNDArray< vector_td<float,4> > *, cuNDArray< vector_td<float,4> > *);

template  boost::shared_ptr<cuNDArray<vector_td<double,1> > > Gadgetron::test_amax<double,1>(cuNDArray< vector_td<double,1> > *, cuNDArray< vector_td<double,1> > *);
template boost::shared_ptr<cuNDArray<vector_td<double,2> > > Gadgetron::test_amax<double,2>(cuNDArray< vector_td<double,2> > *, cuNDArray< vector_td<double,2> > *);
template boost::shared_ptr<cuNDArray<vector_td<double,3> > > Gadgetron::test_amax<double,3>(cuNDArray< vector_td<double,3> > *, cuNDArray< vector_td<double,3> > *);
template boost::shared_ptr<cuNDArray<vector_td<double,4> > > Gadgetron::test_amax<double,4>(cuNDArray< vector_td<double,4> > *, cuNDArray< vector_td<double,4> > *);


template boost::shared_ptr<cuNDArray<vector_td<float,1> > > Gadgetron::test_amax2<float,1>(cuNDArray< vector_td<float,1> > *, float );
template boost::shared_ptr<cuNDArray<vector_td<float,2> > > Gadgetron::test_amax2<float,2>(cuNDArray< vector_td<float,2> > *, float);
template  boost::shared_ptr<cuNDArray<vector_td<float,3> > > Gadgetron::test_amax2<float,3>(cuNDArray< vector_td<float,3> > *, float);
template  boost::shared_ptr<cuNDArray<vector_td<float,4> > > Gadgetron::test_amax2<float,4>(cuNDArray< vector_td<float,4> > *, float);

template  boost::shared_ptr<cuNDArray<vector_td<double,1> > > Gadgetron::test_amax2<double,1>(cuNDArray< vector_td<double,1> > *, double);
template boost::shared_ptr<cuNDArray<vector_td<double,2> > > Gadgetron::test_amax2<double,2>(cuNDArray< vector_td<double,2> > *, double);
template boost::shared_ptr<cuNDArray<vector_td<double,3> > > Gadgetron::test_amax2<double,3>(cuNDArray< vector_td<double,3> > *, double);
template boost::shared_ptr<cuNDArray<vector_td<double,4> > > Gadgetron::test_amax2<double,4>(cuNDArray< vector_td<double,4> > *, double);



template void Gadgetron::vector_fill<float,1>(cuNDArray< vector_td<float,1> > *, vector_td<float,1>);
template void Gadgetron::vector_fill<float,2>(cuNDArray< vector_td<float,2> > *, vector_td<float,2>);
template void Gadgetron::vector_fill<float,3>(cuNDArray< vector_td<float,3> > *, vector_td<float,3>);
template void Gadgetron::vector_fill<float,4>(cuNDArray< vector_td<float,4> > *, vector_td<float,4>);


template void Gadgetron::vector_fill<double,1>(cuNDArray< vector_td<double,1> > *, vector_td<double,1>);
template void Gadgetron::vector_fill<double,2>(cuNDArray< vector_td<double,2> > *, vector_td<double,2>);
template void Gadgetron::vector_fill<double,3>(cuNDArray< vector_td<double,3> > *, vector_td<double,3>);
template void Gadgetron::vector_fill<double,4>(cuNDArray< vector_td<double,4> > *, vector_td<double,4>);
