#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CSI_utils.h"
#include <algorithm>
#include "cudaDeviceManager.h"
#include "complext.h"
#include <stdio.h>
#include "cuNDArray_math.h"
#include "cuNDArray_fileio.h"
#include <numeric>
#include <cmath>

using namespace Gadgetron;


template<class T> static void dft_kernel(complext<T>* __restrict__ kspace, const complext<T>* __restrict__ tspace, T* __restrict__ frequencies, unsigned int spiral_length, unsigned int echoes, unsigned int nfreqs,T dte, T dtt){
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < spiral_length*nfreqs ){
		complext<T> result = 0;
		T frequency = frequencies[idx/spiral_length];
		T time_offset = dtt*(idx%spiral_length);
		unsigned int kpoint = idx%spiral_length;
		for (unsigned int i =0; i < echoes; i++){
                        result += exp(complext<T>(0, -frequency * 2 * 3.141592654F * (dte * i + time_offset))) *
                                  tspace[kpoint + i * spiral_length];
                }
		kspace[idx] = result;
	}
}

template<class T> static void dftH_kernel(const complext<T>* __restrict__ kspace, complext<T>* __restrict__ tspace, T* __restrict__ frequencies, unsigned int spiral_length, unsigned int echoes, unsigned int nfreqs,T dte, T dtt){
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (idx < spiral_length*echoes ){
		complext<T> result = 0;
		unsigned int kpoint = idx%spiral_length;
		T timeshift = dte*(idx/spiral_length)+dtt*kpoint;
		for (unsigned int i =0; i < nfreqs; i++){
                        result += exp(complext<T>(0, frequencies[i] * 2 * 3.141592654F * timeshift)) *
                                  kspace[kpoint + i * spiral_length];
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
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = (elements+threadsPerBlock-1)/threadsPerBlock;
        dpct::dim3 dimGrid(totalBlocksPerGrid);

        std::vector<size_t> dims = tspace->get_dimensions();
	if (totalBlocksPerGrid > cudaDeviceManager::Instance()->max_griddim())
		throw std::runtime_error("CSIOperator: Input dimensions too large");

        /*
        DPCT1026:34: The call to cudaFuncSetCacheConfig was removed because SYCL currently does not support configuring
        shared memory on devices.
        */
        for (int i = 0; i < batches; i++) {

                //size_t batchSize = dimGrid.x*dimBlock.x;

		// Invoke kernel
                /*
                DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit,
                query info::device::max_work_group_size. Adjust the work-group size if needed.
                */
                {
                        auto exp_props =
                            sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

                        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                                auto kspace_get_data_ptr_i_elements_ct0 = kspace->get_data_ptr() + i * elements;
                                auto tspace_get_data_ptr_i_t_elements_ct1 = tspace->get_data_ptr() + i * t_elements;
                                auto frequencies_data_ct2 = frequencies->data();
                                auto dims_ct3 = dims[0];
                                auto dims_ct4 = dims[1];
                                auto frequencies_size_ct5 = frequencies->size();

                                cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

                                cgh.parallel_for<dpct_kernel_name<class dft_kernel_314572, T>>(
                                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), exp_props,
                                    [=](sycl::nd_item<3> item_ct1) {
                                            dft_kernel<T>(kspace_get_data_ptr_i_elements_ct0,
                                                          tspace_get_data_ptr_i_t_elements_ct1, frequencies_data_ct2,
                                                          dims_ct3, dims_ct4, frequencies_size_ct5, dte, dtt);
                                    });
                        });
                }
                CHECK_FOR_CUDA_ERROR();
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
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = (elements+threadsPerBlock-1)/threadsPerBlock;
        dpct::dim3 dimGrid(totalBlocksPerGrid);

        if (totalBlocksPerGrid > cudaDeviceManager::Instance()->max_griddim())
		throw std::runtime_error("CSIOperator: Input dimensions too large");

	//size_t batchSize = dimGrid.x*dimBlock.x;
        /*
        DPCT1026:35: The call to cudaFuncSetCacheConfig was removed because SYCL currently does not support configuring
        shared memory on devices.
        */

        std::vector<size_t> dims = tspace->get_dimensions();

	for (int i =0; i< batches; i++){
		// Invoke kernel
                /*
                DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit,
                query info::device::max_work_group_size. Adjust the work-group size if needed.
                */
                {
                        auto exp_props =
                            sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};

                        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                                auto kspace_get_data_ptr_i_k_elements_ct0 = kspace->get_data_ptr() + i * k_elements;
                                auto tspace_get_data_ptr_i_elements_ct1 = tspace->get_data_ptr() + i * elements;
                                auto frequencies_data_ct2 = frequencies->data();
                                auto dims_ct3 = dims[0];
                                auto dims_ct4 = dims[1];
                                auto frequencies_size_ct5 = frequencies->size();

                                cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

                                cgh.parallel_for<dpct_kernel_name<class dftH_kernel_c0eda0, T>>(
                                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), exp_props,
                                    [=](sycl::nd_item<3> item_ct1) {
                                            dftH_kernel<T>(kspace_get_data_ptr_i_k_elements_ct0,
                                                           tspace_get_data_ptr_i_elements_ct1, frequencies_data_ct2,
                                                           dims_ct3, dims_ct4, frequencies_size_ct5, dte, dtt);
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

template<class T> static void mult_freq_kernel(complext<T>* in_out, complext<T>* freqs, bool conjugate){
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if (conjugate)
                in_out[idx] *= conj(freqs[item_ct1.get_group(1)]);
        else
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
        dpct::dim3 dimBlock(threadsPerBlock);
        int totalBlocksPerGrid = (elements+threadsPerBlock-1)/threadsPerBlock;
        dpct::dim3 dimGrid(totalBlocksPerGrid, dims.back());

        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
                auto exp_props =
                    sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
                dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

                dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                        auto in_out_get_data_ptr_ct0 = in_out->get_data_ptr();
                        auto freqs_get_data_ptr_ct1 = freqs->get_data_ptr();

                        cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());

                        cgh.parallel_for<dpct_kernel_name<class mult_freq_kernel_ae67d6,
                                                          T>>(
                            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), exp_props, [=](sycl::nd_item<3> item_ct1) {
                                    mult_freq_kernel(in_out_get_data_ptr_ct0, freqs_get_data_ptr_ct1, conjugate);
                            });
                });
        }
}




template EXPORTHYPER void Gadgetron::CSI_dft<float>(cuNDArray<float_complext>* kspace,cuNDArray<float_complext>* tspace, cuNDArray<float>* frequencies, float dtt, float dte);
template EXPORTHYPER void Gadgetron::CSI_dftH<float>(cuNDArray<float_complext>* kspace,cuNDArray<float_complext>* tspace, cuNDArray<float>* frequencies, float dtt, float dte);


template EXPORTHYPER boost::shared_ptr<cuNDArray<float_complext> > Gadgetron::calculate_frequency_calibration<float>(cuNDArray<float_complext>* time_track, cuNDArray<float>* frequencies,cuNDArray<float_complext> * csm,float dtt,float dte);
template EXPORTHYPER void Gadgetron::mult_freq<float>(cuNDArray<complext<float> >* in_out, cuNDArray<complext<float> >* freqs, bool conjugate);
