#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDArray.h"
#include <dpct/dpl_utils.hpp>

#include "vector_td.h"
#include "vector_td_utilities.h"
#include "trajectory_utils.h"
using namespace Gadgetron;

struct traj_filter_functor{

	traj_filter_functor(float limit){
		limit_ = limit;
	}
        std::tuple<floatd2, float> operator()(std::tuple<floatd2, float> tup) const {

                floatd2 traj = std::get<0>(tup);
                float dcw = std::get<1>(tup);
                if (sycl::fabs(traj[0]) > limit_ || sycl::fabs(traj[1]) > limit_)
                        dcw = 0;

                return std::tuple<floatd2, float>(traj, dcw);
        }

	float limit_;
};

boost::shared_ptr< cuNDArray<float> > Gadgetron::filter_dcw(cuNDArray<floatd2>* traj, cuNDArray<float>* dcw,float limit){
	cuNDArray<float>* dcw_new = new cuNDArray<float>(*dcw);

        std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()),
                       oneapi::dpl::make_zip_iterator(std::make_tuple(traj->begin(), dcw_new->begin())),
                       oneapi::dpl::make_zip_iterator(std::make_tuple(traj->end(), dcw_new->end())),
                       oneapi::dpl::make_zip_iterator(std::make_tuple(traj->begin(), dcw_new->begin())),
                       traj_filter_functor(limit));

        return boost::shared_ptr<cuNDArray<float> >(dcw_new);


}
