#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CSI_utils.h"
#include <algorithm>
#include "cudaDeviceManager.h"
#include "complext.h"
/* DPCT_ORIG #include <math_constants.h>*/
#include <stdio.h>
#include "cuNDArray_math.h"
#include "cuNDArray_fileio.h"
#include <numeric>
#include <cmath>

using namespace Gadgetron;

/* DPCT_ORIG template<class T> static __global__ void dft_kernel(complext<T>* __restrict__ kspace, const complext<T>*
 * __restrict__ tspace, T* __restrict__ frequencies, unsigned int spiral_length, unsigned int echoes, unsigned int
 * nfreqs,T dte, T dtt){*/
template <class T>
static void dft_kernel(complext<T>* __restrict__ kspace, const complext<T>* __restrict__ tspace,
                       T* __restrict__ frequencies, unsigned int spiral_length, unsigned int echoes,
                       unsigned int nfreqs, T dte, T dtt) {
/* DPCT_ORIG 	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < spiral_length*nfreqs ){
		complext<T> result = 0;
		T frequency = frequencies[idx/spiral_length];
		T time_offset = dtt*(idx%spiral_length);
		unsigned int kpoint = idx%spiral_length;
		for (unsigned int i =0; i < echoes; i++){
/* DPCT_ORIG 			result +=
 * exp(complext<T>(0,-frequency*2*CUDART_PI_F*(dte*i+time_offset)))*tspace[kpoint+i*spiral_length];*/
                        result += exp(complext<T>(0, -frequency * 2 * 3.141592654F * (dte * i + time_offset))) *
                                  tspace[kpoint + i * spiral_length];
                }
		kspace[idx] = result;
	}
}

/* DPCT_ORIG template<class T> static __global__ void dftH_kernel(const complext<T>* __restrict__ kspace, complext<T>*
 * __restrict__ tspace, T* __restrict__ frequencies, unsigned int spiral_length, unsigned int echoes, unsigned int
 * nfreqs,T dte, T dtt){*/
template <class T>
static void dftH_kernel(const complext<T>* __restrict__ kspace, complext<T>* __restrict__ tspace,
                        T* __restrict__ frequencies, unsigned int spiral_length, unsigned int echoes,
                        unsigned int nfreqs, T dte, T dtt) {
/* DPCT_ORIG 	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < spiral_length*echoes ){
		complext<T> result = 0;
		unsigned int kpoint = idx%spiral_length;
		T timeshift = dte*(idx/spiral_length)+dtt*kpoint;
		for (unsigned int i =0; i < nfreqs; i++){
/* DPCT_ORIG 			result +=
 * exp(complext<T>(0,frequencies[i]*2*CUDART_PI_F*timeshift))*kspace[kpoint+i*spiral_length];*/
                        result += exp(complext<T>(0, frequencies[i] * 2 * 3.141592654F * timeshift)) * kspace[kpoint + i * spiral_length];
                }
		tspace[idx] = result;
	}
}



template<class T>
void Gadgetron::CSI_dft(cuNDArray<complext<T> >* kspace,
		cuNDArray<complext<T> >* tspace, cuNDArray<T>* frequencies, T dtt, T dte) {

	size_t elements = kspace->get_size(0)*kspace->get_size(1);
	size_t batches = kspace->get_number_of_elements()/elements;
	size_t t_elements = tspace->get_size(0)*tspace->get_size(1);
	int threadsPerBlock = std::min<int>(elements,cudaDeviceManager::Instance()->max_blockdim());
/* DPCT_ORIG 	dim3 dimBlock(threadsPerBlock);*/
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = (elements+threadsPerBlock-1)/threadsPerBlock;
/* DPCT_ORIG 	dim3 dimGrid(totalBlocksPerGrid);*/
        dpct::dim3 dimGrid(totalBlocksPerGrid);

        std::vector<size_t> dims = tspace->get_dimensions();
	if (totalBlocksPerGrid > cudaDeviceManager::Instance()->max_griddim())
		throw std::runtime_error("CSIOperator: Input dimensions too large");

/* DPCT_ORIG 	cudaFuncSetCacheConfig(dft_kernel<T>,cudaFuncCachePreferL1);*/
        /*
        DPCT1026:211: The call to cudaFuncSetCacheConfig was removed because SYCL currently does not support configuring
        shared memory on devices.
        */
        for (int i = 0; i < batches; i++) {

                //size_t batchSize = dimGrid.x*dimBlock.x;

		// Invoke kernel
/* DPCT_ORIG 		dft_kernel<T><<<dimGrid,
 * dimBlock>>>(kspace->get_data_ptr()+i*elements,tspace->get_data_ptr()+i*t_elements,frequencies->data(),dims[0],dims[1],
 * frequencies->size(),dte,dtt);*/
                /*
                DPCT1049:36: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
                limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
                */
        {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                auto kspace_get_data_ptr_i_elements_ct0 = kspace->get_data_ptr() + i * elements;
                auto tspace_get_data_ptr_i_t_elements_ct1 = tspace->get_data_ptr() + i * t_elements;
                auto frequencies_data_ct2 = frequencies->data();
                auto dims_ct3 = dims[0];
                auto dims_ct4 = dims[1];
                auto frequencies_size_ct5 = frequencies->size();

                cgh.parallel_for<dpct_kernel_name<class dft_kernel_314572, T>>(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                        dft_kernel<T>(kspace_get_data_ptr_i_elements_ct0, tspace_get_data_ptr_i_t_elements_ct1,
                                      frequencies_data_ct2, dims_ct3, dims_ct4, frequencies_size_ct5, dte, dtt);
                    });
            });
        }
                CHECK_FOR_CUDA_ERROR();
/* DPCT_ORIG 		cudaDeviceSynchronize();*/
                dpct::get_current_device().queues_wait_and_throw();
        }

	*kspace /= T(dims[1]);

}

template<class T>
void Gadgetron::CSI_dftH(cuNDArray<complext<T> >* kspace,
		cuNDArray<complext<T> >* tspace, cuNDArray<T>* frequencies, T dtt, T dte) {
	size_t k_elements = kspace->get_size(0)*kspace->get_size(1);
	size_t elements = tspace->get_size(0)*tspace->get_size(1);

	size_t batches = tspace->get_number_of_elements()/elements;
	int threadsPerBlock = std::min<int>(elements,cudaDeviceManager::Instance()->max_blockdim());
/* DPCT_ORIG 	dim3 dimBlock(threadsPerBlock);*/
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = (elements+threadsPerBlock-1)/threadsPerBlock;
/* DPCT_ORIG 	dim3 dimGrid(totalBlocksPerGrid);*/
        dpct::dim3 dimGrid(totalBlocksPerGrid);

        if (totalBlocksPerGrid > cudaDeviceManager::Instance()->max_griddim())
		throw std::runtime_error("CSIOperator: Input dimensions too large");

	//size_t batchSize = dimGrid.x*dimBlock.x;
/* DPCT_ORIG 	cudaFuncSetCacheConfig(dftH_kernel<T>,cudaFuncCachePreferL1);*/
        /*
        DPCT1026:212: The call to cudaFuncSetCacheConfig was removed because SYCL currently does not support configuring
        shared memory on devices.
        */

        std::vector<size_t> dims = tspace->get_dimensions();

	for (int i =0; i< batches; i++){
		// Invoke kernel
/* DPCT_ORIG 		dftH_kernel<T><<<dimGrid,
 * dimBlock>>>(kspace->get_data_ptr()+i*k_elements,tspace->get_data_ptr()+i*elements,frequencies->data(),dims[0],dims[1],
 * frequencies->size(),dte,dtt);*/
                /*
                DPCT1049:37: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
                limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
                */
        {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                auto kspace_get_data_ptr_i_k_elements_ct0 = kspace->get_data_ptr() + i * k_elements;
                auto tspace_get_data_ptr_i_elements_ct1 = tspace->get_data_ptr() + i * elements;
                auto frequencies_data_ct2 = frequencies->data();
                auto dims_ct3 = dims[0];
                auto dims_ct4 = dims[1];
                auto frequencies_size_ct5 = frequencies->size();

                cgh.parallel_for<dpct_kernel_name<class dftH_kernel_c0eda0, T>>(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                        dftH_kernel<T>(kspace_get_data_ptr_i_k_elements_ct0, tspace_get_data_ptr_i_elements_ct1,
                                       frequencies_data_ct2, dims_ct3, dims_ct4, frequencies_size_ct5, dte, dtt);
                    });
            });
        }
                CHECK_FOR_CUDA_ERROR();
	}
	*tspace /= T(dims[1]);
}

template<class T>
boost::shared_ptr<cuNDArray<complext<T> > > Gadgetron::calculate_frequency_calibration(cuNDArray<complext<T> >* time_track, cuNDArray<T>* frequencies,cuNDArray<complext<T> > * csm,T dtt,T dte){
	std::vector<size_t> out_dims;
	out_dims.push_back(frequencies->size());
	out_dims.push_back(1);

	cuNDArray<complext<T> >* time2 = time_track;
	if (csm){
		std::vector<size_t> csm_dims = csm->get_dimensions();
		int coils = csm_dims.back();
		csm_dims.pop_back();

		std::vector<size_t> time_dims = time_track->get_dimensions();

		if (time_dims.back() != coils)
			throw std::runtime_error("Number of coils in time data does not match number of coils in CSM");


		time_dims.back() = time_dims.front();
		time_dims.front() = 1;
		time_dims.push_back(coils);

		time2 = new cuNDArray<complext<T> >(time_dims, time_track->get_data_ptr());
		out_dims.push_back(coils);
	}

	boost::shared_ptr<cuNDArray<complext<T > > > result(new cuNDArray<complext<T> >(out_dims));
	clear(result.get());

	CSI_dft(result.get(),time2,frequencies,float(0),dtt);

	if (csm)
		delete time2;
	return result;



}

/* DPCT_ORIG template<class T> static __global__ void mult_freq_kernel(complext<T>* in_out, complext<T>* freqs, bool
 * conjugate){*/
template <class T> static void mult_freq_kernel(complext<T>* in_out, complext<T>* freqs, bool conjugate) {
/* DPCT_ORIG 	const int idx =  blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (conjugate)
/* DPCT_ORIG 		in_out[idx] *= conj(freqs[blockIdx.y]);*/
                in_out[idx] *= conj(freqs[item_ct1.get_group(1)]);
        else
/* DPCT_ORIG 		in_out[idx] *= freqs[blockIdx.y];*/
                in_out[idx] *= freqs[item_ct1.get_group(1)];
}
template< class T>
void Gadgetron::mult_freq(cuNDArray<complext<T> >* in_out, cuNDArray<complext<T> >* freqs, bool conjugate){

	std::vector<size_t> dims = in_out->get_dimensions();

	if (dims.back() != freqs->get_number_of_elements()){
		throw std::runtime_error("Input image dimensions do not match frequencies");
	}

	size_t elements = in_out->get_number_of_elements()/dims.back();
	int threadsPerBlock = std::min<size_t>(elements,cudaDeviceManager::Instance()->max_blockdim());
/* DPCT_ORIG 	dim3 dimBlock(threadsPerBlock);*/
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = (elements+threadsPerBlock-1)/threadsPerBlock;
/* DPCT_ORIG 	dim3 dimGrid(totalBlocksPerGrid,dims.back());*/
        dpct::dim3 dimGrid(totalBlocksPerGrid, dims.back());

/* DPCT_ORIG 	mult_freq_kernel<<<dimGrid, dimBlock>>>(in_out->get_data_ptr(),freqs->get_data_ptr(),conjugate);*/
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto in_out_get_data_ptr_ct0 = in_out->get_data_ptr();
            auto freqs_get_data_ptr_ct1 = freqs->get_data_ptr();

            /*
            DPCT1050:302: The template argument of the dpct_kernel_name could not be deduced. You need to update this
            code.
            */
            cgh.parallel_for(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                    mult_freq_kernel(in_out_get_data_ptr_ct0, freqs_get_data_ptr_ct1, conjugate);
                });
        });
    }
}




template EXPORTHYPER void Gadgetron::CSI_dft<float>(cuNDArray<float_complext>* kspace,cuNDArray<float_complext>* tspace, cuNDArray<float>* frequencies, float dtt, float dte);
template EXPORTHYPER void Gadgetron::CSI_dftH<float>(cuNDArray<float_complext>* kspace,cuNDArray<float_complext>* tspace, cuNDArray<float>* frequencies, float dtt, float dte);


template EXPORTHYPER boost::shared_ptr<cuNDArray<float_complext> > Gadgetron::calculate_frequency_calibration<float>(cuNDArray<float_complext>* time_track, cuNDArray<float>* frequencies,cuNDArray<float_complext> * csm,float dtt,float dte);
template EXPORTHYPER void Gadgetron::mult_freq<float>(cuNDArray<complext<float> >* in_out, cuNDArray<complext<float> >* freqs, bool conjugate);
