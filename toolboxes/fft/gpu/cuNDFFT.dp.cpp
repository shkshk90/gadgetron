#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDFFT.h"
#include "cudaDeviceManager.h"
#include <cmath>

using namespace Gadgetron;

/* DPCT_ORIG template<class T> __global__ void timeswitch_kernel(T* data, int dimsize, int batchsize, size_t
 * nelements){*/
template <class T> void timeswitch_kernel(T* data, int dimsize, int batchsize, size_t nelements) {
/* DPCT_ORIG 	int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < nelements){
		int index = (idx/batchsize)%dimsize;
		if (index & 1) //Check if number is odd
			data[idx] *= -1;
	}
}

/* DPCT_ORIG template<class T> __global__ void timeswitch_kernel1D(T* data, size_t nelements){*/
template <class T> void timeswitch_kernel1D(T* data, size_t nelements) {

/* DPCT_ORIG 	int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < nelements){
/* DPCT_ORIG 		data[idx] *= (-int(threadIdx.x & 1)*2+1); */
                data[idx] *= (-int(item_ct1.get_local_id(2) & 1) * 2 + 1); // Multiply by -1 if x  is odd
        }
}

/* DPCT_ORIG template<class T> __global__ void timeswitch_kernel2D(T* data, size_t nelements){*/
template <class T> void timeswitch_kernel2D(T* data, size_t nelements) {

/* DPCT_ORIG 	int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < nelements)
/* DPCT_ORIG 		data[idx] *= (-int(threadIdx.x & 1)*2+1)*int(-(blockIdx.x & 1)*2+1); */
                data[idx] *= (-int(item_ct1.get_local_id(2) & 1) * 2 + 1) *
                             int(-(item_ct1.get_group(2) & 1) * 2 +
                                 1); // Multiply by -1 if x or y coordinate is odd, but not if both are
}

/* DPCT_ORIG template<class T> __global__ void timeswitch_kernel3D(T* data, size_t nelements){*/
template <class T> void timeswitch_kernel3D(T* data, size_t nelements) {
/* DPCT_ORIG 	int idx = ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int idx = ((item_ct1.get_group(0) * item_ct1.get_group_range(1) + item_ct1.get_group(1)) *
                       item_ct1.get_group_range(2) +
                   item_ct1.get_group(2)) *
                      item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if (idx < nelements)
/* DPCT_ORIG 		data[idx] *= (-int(threadIdx.x & 1)*2+1)*(-int(blockIdx.x & 1)*2+1)*(-int(blockIdx.y & 1)*2+1);
 */
                data[idx] *= (-int(item_ct1.get_local_id(2) & 1) * 2 + 1) * (-int(item_ct1.get_group(2) & 1) * 2 + 1) *
                             (-int(item_ct1.get_group(1) & 1) * 2 +
                              1); // Multiply by -1 if x or y coordinate is odd, but not if both are
}

template<class T> void Gadgetron::timeswitch1D(cuNDArray<complext<T> >* inout){
	if (inout->get_size(0) > cudaDeviceManager::Instance()->max_blockdim())
		return timeswitch(inout,0);
/* DPCT_ORIG 	dim3 dimBlock(inout->get_size(0));*/
        dpct::dim3 dimBlock(inout->get_size(0));
        size_t max_grid = cudaDeviceManager::Instance()->max_griddim();
	size_t nelements = inout->get_number_of_elements();
	size_t gridX = std::max(std::min(nelements/dimBlock.x,max_grid),size_t(1));
	size_t gridY = std::max(size_t(1),nelements/(gridX*dimBlock.x));
/* DPCT_ORIG 	dim3 dimGrid(gridX,gridY);*/
        dpct::dim3 dimGrid(gridX, gridY);
/* DPCT_ORIG 	timeswitch_kernel1D<<<dimGrid,dimBlock>>>(inout->get_data_ptr(),nelements);*/
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
        auto inout_get_data_ptr_ct0 = inout->get_data_ptr();

        /*
        DPCT1050:292: The template argument of the dpct_kernel_name could not be deduced. You need to update this
        code.
        */
        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                timeswitch_kernel1D(inout_get_data_ptr_ct0, nelements);
            });
    });
}

template<class T> void Gadgetron::timeswitch2D(cuNDArray<complext<T> >* inout){
	if (inout->get_size(0) > cudaDeviceManager::Instance()->max_blockdim()){
		timeswitch(inout,0);
		timeswitch(inout,1);
		return;
	}
/* DPCT_ORIG 	dim3 dimBlock(inout->get_size(0));*/
        dpct::dim3 dimBlock(inout->get_size(0));
        size_t max_grid = cudaDeviceManager::Instance()->max_griddim();
	size_t nelements = inout->get_number_of_elements();
	size_t gridX = inout->get_size(1);
	size_t gridY = std::max(size_t(1),nelements/(gridX*dimBlock.x));
/* DPCT_ORIG 	dim3 dimGrid(gridX,gridY);*/
        dpct::dim3 dimGrid(gridX, gridY);
/* DPCT_ORIG 	timeswitch_kernel2D<<<dimGrid,dimBlock>>>(inout->get_data_ptr(),nelements);*/
        /*
        DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
        auto inout_get_data_ptr_ct0 = inout->get_data_ptr();

        /*
        DPCT1050:293: The template argument of the dpct_kernel_name could not be deduced. You need to update this
        code.
        */
        cgh.parallel_for
        // <dpct_kernel_name<class timeswitch_kernel2D_cbe240, dpct_placeholder /*Fix the type mannually*/>>
            (
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                timeswitch_kernel2D(inout_get_data_ptr_ct0, nelements);
            });
    });
}


template<class T> void Gadgetron::timeswitch3D(cuNDArray<complext<T> >* inout){
	if (inout->get_size(0) > cudaDeviceManager::Instance()->max_blockdim()){
		timeswitch(inout,0);
		timeswitch(inout,1);
		timeswitch(inout,2);
		return;
	}
/* DPCT_ORIG 	dim3 dimBlock(inout->get_size(0));*/
        dpct::dim3 dimBlock(inout->get_size(0));
        size_t max_grid = cudaDeviceManager::Instance()->max_griddim();
	size_t nelements = inout->get_number_of_elements();
	size_t gridX = inout->get_size(1);
	size_t gridY = inout->get_size(2);
	size_t gridZ = std::max(size_t(1),nelements/(gridX*dimBlock.x*gridY));
/* DPCT_ORIG 	dim3 dimGrid(gridX,gridY,gridZ);*/
        dpct::dim3 dimGrid(gridX, gridY, gridZ);
/* DPCT_ORIG 	timeswitch_kernel3D<<<dimGrid,dimBlock>>>(inout->get_data_ptr(),nelements);*/
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
        auto inout_get_data_ptr_ct0 = inout->get_data_ptr();

        /*
        DPCT1050:294: The template argument of the dpct_kernel_name could not be deduced. You need to update this
        code.
        */
        cgh.parallel_for<
            dpct_kernel_name<class timeswitch_kernel3D_cba933, dpct_placeholder /*Fix the type mannually*/>>(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                timeswitch_kernel3D(inout_get_data_ptr_ct0, nelements);
            });
    });
}

template<class T> void Gadgetron::timeswitch(cuNDArray<complext<T> >* inout, int dim_to_transform){


	size_t batchsize = 1;
	for (int i = 0; i < dim_to_transform; i++)
		batchsize *= inout->get_size(i);

	size_t dimsize = inout->get_size(dim_to_transform);

	size_t nelements = inout->get_number_of_elements();

	size_t max_block = cudaDeviceManager::Instance()->max_blockdim();
/* DPCT_ORIG 	dim3 dimBlock(std::min(max_block,nelements));*/
        dpct::dim3 dimBlock(std::min(max_block, nelements));

        size_t max_grid = cudaDeviceManager::Instance()->max_griddim();
	size_t gridX = std::max(std::min(nelements/dimBlock.x,max_grid),size_t(1));
	size_t gridY = std::max(size_t(1),nelements/(gridX*dimBlock.x));

/* DPCT_ORIG 	dim3 dimGrid(gridX,gridY);*/
        dpct::dim3 dimGrid(gridX, gridY);

/* DPCT_ORIG 	timeswitch_kernel<<<dimGrid,dimBlock>>>(inout->get_data_ptr(),dimsize,batchsize,nelements);*/
        /*
        DPCT1049:35: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
        auto inout_get_data_ptr_ct0 = inout->get_data_ptr();

        /*
        DPCT1050:295: The template argument of the dpct_kernel_name could not be deduced. You need to update this
        code.
        */
        cgh.parallel_for<dpct_kernel_name<class timeswitch_kernel_999838, dpct_placeholder /*Fix the type mannually*/>>(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                timeswitch_kernel(inout_get_data_ptr_ct0, dimsize, batchsize, nelements);
            });
    });
}



template EXPORTGPUFFT void Gadgetron::timeswitch<float>(cuNDArray<float_complext>*, int);
template EXPORTGPUFFT void Gadgetron::timeswitch<double>(cuNDArray<double_complext>*, int);

template EXPORTGPUFFT void Gadgetron::timeswitch1D<float>(cuNDArray<float_complext>*);
template EXPORTGPUFFT void Gadgetron::timeswitch1D<double>(cuNDArray<double_complext>*);

template EXPORTGPUFFT void Gadgetron::timeswitch2D<float>(cuNDArray<float_complext>*);
template EXPORTGPUFFT void Gadgetron::timeswitch2D<double>(cuNDArray<double_complext>*);

template EXPORTGPUFFT void Gadgetron::timeswitch3D<float>(cuNDArray<float_complext>*);
template EXPORTGPUFFT void Gadgetron::timeswitch3D<double>(cuNDArray<double_complext>*);
