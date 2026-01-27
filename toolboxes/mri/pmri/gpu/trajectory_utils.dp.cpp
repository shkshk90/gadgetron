#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDArray.h"
#include <dpct/dpl_utils.hpp>

/* DPCT_ORIG #include <thrust/iterator/zip_iterator.h>*/
#include "vector_td.h"
#include "vector_td_utilities.h"
#include "trajectory_utils.h"
using namespace Gadgetron;

struct traj_filter_functor{

	traj_filter_functor(float limit){
		limit_ = limit;
	}
/* DPCT_ORIG 	__device__  thrust::tuple<floatd2,float> operator()(thrust::tuple<floatd2,float> tup){*/
        std::tuple<floatd2, float> operator()(std::tuple<floatd2, float> tup) const {

/* DPCT_ORIG 		floatd2 traj = thrust::get<0>(tup);*/
                floatd2 traj = std::get<0>(tup);
/* DPCT_ORIG 		float dcw = thrust::get<1>(tup);*/
                float dcw = std::get<1>(tup);
/* DPCT_ORIG 		if ( abs(traj[0]) > limit_ || abs(traj[1]) > limit_)*/
                if (sycl::fabs(traj[0]) > limit_ || sycl::fabs(traj[1]) > limit_)
                        dcw = 0;

/* DPCT_ORIG 		return thrust::tuple<floatd2,float>(traj,dcw);*/
                return std::tuple<floatd2, float>(traj, dcw);
        }

	float limit_;
};

boost::shared_ptr< cuNDArray<float> > Gadgetron::filter_dcw(cuNDArray<floatd2>* traj, cuNDArray<float>* dcw,float limit){
	cuNDArray<float>* dcw_new = new cuNDArray<float>(*dcw);

/* DPCT_ORIG 	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(traj->begin(),dcw_new->begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(traj->end(),dcw_new->end())),
                        thrust::make_zip_iterator(thrust::make_tuple(traj->begin(),dcw_new->begin())),
                        traj_filter_functor(limit));*/
        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()),
                       oneapi::dpl::make_zip_iterator(std::make_tuple(traj->begin(), dcw_new->begin())),
                       oneapi::dpl::make_zip_iterator(std::make_tuple(traj->end(), dcw_new->end())),
                       oneapi::dpl::make_zip_iterator(std::make_tuple(traj->begin(), dcw_new->begin())),
                       traj_filter_functor(limit));

        return boost::shared_ptr<cuNDArray<float> >(dcw_new);


}
