#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuHaarWaveletOperator.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_blas.h"
#include "cuNDArray_utils.h"
#include "vector_td_utilities.h"
#include "vector_td_io.h"
#include "complext.h"
#include <iostream>
#include "check_CUDA.h"
#include "cudaDeviceManager.h"
#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace Gadgetron;


// Template Power function
template<unsigned int i, unsigned int j>
struct Pow
{
	enum { Value = i*Pow<i,j-1>::Value};
};

template <unsigned int i>
struct Pow<i,1>
{
	enum { Value = i};
};


template<class T> struct Haar{
/* DPCT_ORIG 	static inline __device__ void lift(T& x1,T& x2){*/
        static inline void lift(T& x1, T& x2) {
                T s0 = x1;
		T d0 = x2;
		x1 = (s0+d0)/T(2); //s_l
		x2 = s0-d0; //d_l
	}

/* DPCT_ORIG 	static inline __device__ void sink(T& x1,T& x2){*/
        static inline void sink(T& x1, T& x2) {
                T s0 = x1;
		T d0 = x2;
		x1 = s0+d0/T(2);
		x2 = s0-d0/T(2);
	}
};


template<class T, unsigned int D,class wave, unsigned int N>  struct recWave{

/* DPCT_ORIG 	static inline __device__ void loadData(T* elements, T* data, const vector_td<int,D>& dims){*/
        static inline void loadData(T* elements, T* data, const vector_td<int, D>& dims) {
                int offset = 1;
		for (int i = 0; i < N; i++)
			offset *= dims[N-i-1];

		recWave<T,D,wave,N-1>::loadData(elements,data,dims);
		recWave<T,D,wave,N-1>::loadData(elements+Pow<2,N>::Value,data+offset,dims);
	}

/* DPCT_ORIG 	static inline __device__ void predict(T* elements){*/
        static inline void predict(T* elements) {
                recWave<T,D,wave,N-1>::predict(elements);
		recWave<T,D,wave,N-1>::predict(elements+Pow<2,N>::Value);
		for (int i = 0; i < Pow<2,N>::Value; i++){
			wave::lift(elements[i],elements[Pow<2,N>::Value+i]);
		}
	}

/* DPCT_ORIG 	static inline __device__ void ipredict(T* elements){*/
        static inline void ipredict(T* elements) {
                for (int i = 0; i < Pow<2,N>::Value; i++){
			wave::sink(elements[i],elements[Pow<2,N>::Value+i]);
		}
		recWave<T,D,wave,N-1>::ipredict(elements);
		recWave<T,D,wave,N-1>::ipredict(elements+Pow<2,N>::Value);
	}

/* DPCT_ORIG 	static inline __device__ void saveData(T* elements, T* data, const vector_td<int,D>& dims){*/
        static inline void saveData(T* elements, T* data, const vector_td<int, D>& dims) {
                int offset = 1;
		for (int i = 0; i < N; i++)
			offset *= dims[N-i-1];

		recWave<T,D,wave,N-1>::saveData(elements,data,dims);
		recWave<T,D,wave,N-1>::saveData(elements+Pow<2,N>::Value,data+offset,dims);
	}


};

template<class T, unsigned int D, class wave> struct recWave<T,D,wave,0>{

/* DPCT_ORIG 	static inline __device__ void loadData(T* elements, T* data, const vector_td<int,D>& dims){*/
        static inline void loadData(T* elements, T* data, const vector_td<int, D>& dims) {
                elements[0] = data[0];
		elements[1] = data[1];
	}

/* DPCT_ORIG 	static inline __device__ void predict(T* elements){*/
        static inline void predict(T* elements) {
                wave::lift(elements[0],elements[1]);
	}

/* DPCT_ORIG 	static inline __device__ void ipredict(T* elements){*/
        static inline void ipredict(T* elements) {
                wave::sink(elements[0],elements[1]);
	}
/* DPCT_ORIG 	static inline __device__ void saveData(T* elements, T* data, const vector_td<int,D>& dims){*/
        static inline void saveData(T* elements, T* data, const vector_td<int, D>& dims) {
                data[0] = elements[0];
		data[1] = elements[1];
	}
};

/* DPCT_ORIG template<class T, unsigned int D, class wave> __global__ void haarKernel(T* in, T* out, const
 * vector_td<int,D> dims){*/
/*
DPCT1110:24: The total declared local variable size in device function haarKernel exceeds 128 bytes and may cause high
register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or
use smaller sub-group size to avoid high register pressure.
*/
template <class T, unsigned int D, class wave> void haarKernel(T* in, T* out, const vector_td<int, D> dims) {

        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        T elements[Pow<2, D>::Value];
        int newsize = prod(dims)/Pow<2,D>::Value;
	const vector_td<int,D> dims2 = dims/2;

/* DPCT_ORIG 	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if( idx < newsize ){
		vector_td<int,D> co = idx_to_co<int,D>(idx,dims2);
		co *= 2;
		recWave<T,D,wave,D-1>::loadData(elements,in+co_to_idx<int, D>(co,dims),dims);
		recWave<T,D,wave,D-1>::predict(elements);

		for (int i = 0; i < Pow<2,D>::Value; i++){
			out[i*newsize+idx] = elements[i];
		}

	}
}

/* DPCT_ORIG template<class T, unsigned int D, class wave> __global__ void inv_haarKernel(T* in, T* out, const
 * vector_td<int,D> dims){*/
/*
DPCT1110:25: The total declared local variable size in device function inv_haarKernel exceeds 128 bytes and may cause
high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
template <class T, unsigned int D, class wave> void inv_haarKernel(T* in, T* out, const vector_td<int, D> dims) {

        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        T elements[Pow<2, D>::Value];

        const vector_td<int,D> dims2 = dims/2;
	int oldsize = prod(dims2);
/* DPCT_ORIG 	const int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;*/
        const int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        if( idx < oldsize ){

		for (int i = 0; i < Pow<2,D>::Value; i++){
			elements[i] = in[i*oldsize+idx] ;
		}
		recWave<T,D,wave,D-1>::ipredict(elements);
		vector_td<int,D> co = idx_to_co<int,D>(idx,dims2);
		co *= 2;

		recWave<T,D,wave,D-1>::saveData(elements,out+co_to_idx<int,D>(co,dims),dims);



	}
}

static inline bool isPowerOfTwo (unsigned int x)
{
	return ((x != 0) && ((x & (~x + 1)) == x));
}

template<typename T> inline T next_power2(T value)
{
	--value;
	for(size_t i = 1; i < sizeof(T) * CHAR_BIT; i*=2)
		value |= value >> i;
	return value+1;
}


template<class T, unsigned int D> void cuHaarWaveletOperator<T,D>::set_domain_dimensions(const std::vector<size_t>& dims){

	linearOperator<cuNDArray<T> >::set_domain_dimensions(dims);
	std::vector<size_t> newdims;
	for (int i = 0; i < dims.size(); i++){
		if (isPowerOfTwo(dims.at(i)))
			newdims.push_back(dims.at(i));
		else
			newdims.push_back(next_power2(dims.at(i)));
	}

	linearOperator<cuNDArray<T> >::set_codomain_dimensions(newdims);
}

template<class T, unsigned int D> void cuHaarWaveletOperator<T,D>::mult_M(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate ){
	if (! in->dimensions_equal(this->get_domain_dimensions()))
		throw std::runtime_error("cuHaarWaveletOperator::mult_M: size of input array does not match operator domain size.");
	if (! out->dimensions_equal(this->get_codomain_dimensions()))
		throw std::runtime_error("cuHaarWaveletOperator::mult_M: size of output array does not match operator codomain size.");


	cuNDArray<T>* tmp_in = new cuNDArray<T>(this->get_codomain_dimensions());

	if (in->dimensions_equal(tmp_in))
		*tmp_in = *in;
	else
		pad<T,D>(*in,*tmp_in);


	cuNDArray<T>* tmp_out;
	if (accumulate)
		tmp_out =new cuNDArray<T>(tmp_in->get_dimensions());
	else
		tmp_out = out;

	typename intd<D>::Type dims = vector_td<int,D>( from_std_vector<size_t,D>(tmp_in->get_dimensions()));

	typename intd<D>::Type dims2;
	for (int i = 0; i < D; i++) dims2[i] = 2;

	while(dims >= dims2){
		int elements=  prod(dims)/Pow<2,D>::Value;
		int threadsPerBlock =std::min(elements,256);
/* DPCT_ORIG 		dim3 dimBlock( threadsPerBlock);*/
                dpct::dim3 dimBlock(threadsPerBlock);
                int totalBlocksPerGrid = std::max(1,elements/threadsPerBlock);
/* DPCT_ORIG 		dim3 dimGrid(totalBlocksPerGrid);*/
                dpct::dim3 dimGrid(totalBlocksPerGrid);
/* DPCT_ORIG 		haarKernel<T,D,Haar<T>
 * ><<<dimGrid,dimBlock>>>(tmp_in->get_data_ptr(),tmp_out->get_data_ptr(),dims);*/
                /*
                DPCT1049:26: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
                limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
                */
        {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                auto tmp_in_get_data_ptr_ct0 = tmp_in->get_data_ptr();
                auto tmp_out_get_data_ptr_ct1 = tmp_out->get_data_ptr();

                cgh.parallel_for<dpct_kernel_name<class haarKernel_c2ea92, T, dpct_kernel_scalar<D>, Haar<T>>>(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                        haarKernel<T, D, Haar<T>>(tmp_in_get_data_ptr_ct0, tmp_out_get_data_ptr_ct1, dims);
                    });
            });
        }
                CHECK_FOR_CUDA_ERROR();
		dims /= 2;
		*tmp_in = *tmp_out;
	}
	delete tmp_in;

	dims2 /= 2;
	if (dims != dims2){
		std::vector<size_t> sdim = to_std_vector(vector_td<size_t,D>(dims));
		cuNDArray<T> smallArray(sdim,tmp_out->get_data_ptr());
		smallArray.squeeze();
		cuNDArray<T> smallTmp(smallArray);
		linearOperator<cuNDArray<T> >* smallWave;
		switch(smallArray.get_number_of_dimensions()){
		case 1:
			smallWave = new cuHaarWaveletOperator<T,1>;
			break;
		case 2:
			smallWave = new cuHaarWaveletOperator<T,2>;
			break;
		case 3:
			smallWave = new cuHaarWaveletOperator<T,3>;
			break;
		default:
			throw std::logic_error("cuHaarWaveletOperator::mult_M: Illegal number of input dimensions given");
		}
		smallWave->set_domain_dimensions(smallArray.get_dimensions());
		smallWave->mult_M(&smallTmp,&smallArray,false);
		delete smallWave;
	}

	if (accumulate){
		*out += *tmp_out;
		delete tmp_out;
	}


}


template<class T, unsigned int D> void cuHaarWaveletOperator<T,D>::mult_MH(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate ){


	if (! out->dimensions_equal(this->get_domain_dimensions()))
		throw std::runtime_error("cuHaarWaveletOperator::mult_MH: size of output array does not match operator domain size.");
	if (! in->dimensions_equal(this->get_codomain_dimensions()))
		throw std::runtime_error("cuHaarWaveletOperator::mult_MH: size of input array does not match operator codomain size.");
	cuNDArray<T>* tmp_out = new cuNDArray<T>(*in);

	cuNDArray<T>* tmp_in = new cuNDArray<T>(*in);


	const typename intd<D>::Type dims = vector_td<int,D>( from_std_vector<size_t,D>(in->get_dimensions()));

	typename intd<D>::Type cur_dims = dims/min(dims);


	if (prod(cur_dims) > 1){
		std::vector<size_t> sdim = to_std_vector(vector_td<size_t,D>(cur_dims));
		cuNDArray<T> smallIn(sdim,tmp_in->get_data_ptr());
		smallIn.squeeze();
		cuNDArray<T> smallOut(sdim,tmp_out->get_data_ptr());
		smallOut.squeeze();
		linearOperator<cuNDArray<T> >* smallWave;

		switch(smallIn.get_number_of_dimensions()){
		case 1:
			smallWave = new cuHaarWaveletOperator<T,1>;
			break;
		case 2:
			smallWave = new cuHaarWaveletOperator<T,2>;
			break;
		case 3:
			smallWave = new cuHaarWaveletOperator<T,3>;
			break;
		default:
			throw std::logic_error("cuHaarWaveletOperator::mult_M: 5D wavelets are currently considered overly ambitious.");
		}
		smallWave->set_domain_dimensions(smallIn.get_dimensions());
		smallWave->mult_MH(&smallIn,&smallOut,false);
		smallIn = smallOut;
		delete smallWave;
	}

	while(cur_dims <= dims){
		int elements = prod(cur_dims*2)/Pow<2,D>::Value;
		int threadsPerBlock =std::min(elements,cudaDeviceManager::Instance()->max_blockdim());
/* DPCT_ORIG 		dim3 dimBlock( threadsPerBlock);*/
                dpct::dim3 dimBlock(threadsPerBlock);
                int totalBlocksPerGrid = std::max(1,elements/cudaDeviceManager::Instance()->max_blockdim());
/* DPCT_ORIG 		dim3 dimGrid(totalBlocksPerGrid);*/
                dpct::dim3 dimGrid(totalBlocksPerGrid);

/* DPCT_ORIG 		inv_haarKernel<T,D,Haar<T>
 * ><<<dimGrid,dimBlock>>>(tmp_in->get_data_ptr(),tmp_out->get_data_ptr(),cur_dims);*/
                /*
                DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
                limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
                */
        {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                auto tmp_in_get_data_ptr_ct0 = tmp_in->get_data_ptr();
                auto tmp_out_get_data_ptr_ct1 = tmp_out->get_data_ptr();

                cgh.parallel_for<dpct_kernel_name<class inv_haarKernel_67b95c, T, dpct_kernel_scalar<D>, Haar<T>>>(
                    sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item_ct1) {
                        inv_haarKernel<T, D, Haar<T>>(tmp_in_get_data_ptr_ct0, tmp_out_get_data_ptr_ct1, cur_dims);
                    });
            });
        }
                CHECK_FOR_CUDA_ERROR();
		cur_dims *= 2;
		*tmp_in = *tmp_out;
	}

	if (!in->dimensions_equal(this->domain_dims_)){
		delete tmp_in;
		tmp_in = new cuNDArray<T>(this->domain_dims_);
		vector_td<size_t,D> offset;
		for (int i = 0; i < D; i++ ) offset[i] = (this->codomain_dims_[i]-this->domain_dims_[i])/2;
		throw std::runtime_error("cuHaarWaveletOperator::crop error: Size was never implemented.");
		//crop<T,D>(offset,*tmp_out,*tmp_in);
	}

	if (accumulate){
		*out += *tmp_in;
	} else {
		*out = *tmp_in;
	}
	delete tmp_in;
	delete tmp_out;

}

template class  Gadgetron::cuHaarWaveletOperator<float,1>;
template class  Gadgetron::cuHaarWaveletOperator<float,2>;
template class  Gadgetron::cuHaarWaveletOperator<float,3>;
template class  Gadgetron::cuHaarWaveletOperator<float,4>;

template class  Gadgetron::cuHaarWaveletOperator<double,1>;
template class  Gadgetron::cuHaarWaveletOperator<double,2>;
template class  Gadgetron::cuHaarWaveletOperator<double,3>;
template class  Gadgetron::cuHaarWaveletOperator<double,4>;

template class  Gadgetron::cuHaarWaveletOperator<float_complext,1>;
template class  Gadgetron::cuHaarWaveletOperator<float_complext,2>;
template class  Gadgetron::cuHaarWaveletOperator<float_complext,3>;
template class  Gadgetron::cuHaarWaveletOperator<float_complext,4>;

template class  Gadgetron::cuHaarWaveletOperator<double_complext,1>;
template class  Gadgetron::cuHaarWaveletOperator<double_complext,2>;
template class  Gadgetron::cuHaarWaveletOperator<double_complext,3>;
template class  Gadgetron::cuHaarWaveletOperator<double_complext,4>;


