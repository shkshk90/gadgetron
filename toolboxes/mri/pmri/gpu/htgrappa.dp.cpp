#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "htgrappa.h"
#include "cuNDArray_elemwise.h"
#include "cuNDFFT.h"
#include "GadgetronTimer.h"
#include "GPUTimer.h"
#include "cuNDArray_elemwise.h"
#include "CUBLASContextProvider.h"
#include "cpu/hoNDArray_fileio.h"
#include <dpct/blas_utils.hpp>

/* DPCT_ORIG #include <cublas_v2.h>*/
// #include <cula_lapack_device.h>
#include <iostream>
#include <cmath>

namespace Gadgetron {

/* DPCT_ORIG   static int2 vec_to_int2(std::vector<unsigned int> vec)*/
  static sycl::int2 vec_to_int2(std::vector<unsigned int> vec)
  {
/* DPCT_ORIG     int2 ret; ret.x = 0; ret.y = 0;*/
    sycl::int2 ret; ret.x() = 0; ret.y() = 0;
    if (vec.size() < 2) {
      GDEBUG_STREAM("vec_to_uint2 dimensions of vector too small" << std::endl);
      return ret;
    }

/* DPCT_ORIG     ret.x = vec[0]; ret.y = vec[1];*/
    ret.x() = vec[0]; ret.y() = vec[1];
    return ret;
  }

  template <class T> static int write_cuNDArray_to_disk(cuNDArray<T>* a, const char* filename)
  {
    boost::shared_ptr< hoNDArray<T> > host = a->to_host();
    write_nd_array<complext<float> >(host.get(), filename);
    return 0;
  }

/* DPCT_ORIG   template <class T> __global__ void form_grappa_system_matrix_kernel_2d(const T* __restrict__ ref_data,
                                                                         int2 dims,
                                                                         int source_coils,
                                                                         int target_coils,
                                                                         int2 ros,
                                                                         int2 ros_offset,
                                                                         int2 kernel_size,
                                                                         int acceleration_factor,
                                                                         int set_number,
                                                                         T* __restrict__ out_matrix,
                                                                         T* __restrict__ b)*/
  template <class T>
  void form_grappa_system_matrix_kernel_2d(const T* __restrict__ ref_data, sycl::int2 dims, int source_coils,
                                           int target_coils, sycl::int2 ros, sycl::int2 ros_offset,
                                           sycl::int2 kernel_size, int acceleration_factor, int set_number,
                                           T* __restrict__ out_matrix, T* __restrict__ b)
  {
/* DPCT_ORIG     long idx_in = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    long idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
/* DPCT_ORIG     int klocations = ros.x*ros.y;*/
    int klocations = ros.x() * ros.y();
/* DPCT_ORIG     int image_elements = dims.x*dims.y;*/
    int image_elements = dims.x() * dims.y();
    //int coefficients = kernel_size.x*kernel_size.y*coils;
    if (idx_in < klocations) {
      //unsigned int y = idx_in/ros.x;
      //unsigned int x = idx_in - y*ros.x;
/* DPCT_ORIG       unsigned int x = idx_in/ros.y;*/
      unsigned int x = idx_in / ros.y();
/* DPCT_ORIG       unsigned int y = idx_in - x*ros.y;*/
      unsigned int y = idx_in - x * ros.y();
      unsigned int idx_ref = 0;
      unsigned int coeff_counter = 0;

/* DPCT_ORIG       int kernel_size_x = kernel_size.x;*/
      int kernel_size_x = kernel_size.x();
/* DPCT_ORIG       int kernel_size_y = kernel_size.y;*/
      int kernel_size_y = kernel_size.y();

      for (int c = 0; c < source_coils; c++) {
        for (int ky = -((kernel_size_y*acceleration_factor)>>1)+set_number+1;
             ky < ((kernel_size_y*acceleration_factor+1)>>1); ky+=acceleration_factor) {
          for (int kx = -(kernel_size_x>>1); kx < ((kernel_size_x+1)>>1); kx++) {
/* DPCT_ORIG             idx_ref = c*image_elements + x+kx+ros_offset.x + (y+ky+ros_offset.y)*dims.x;*/
            idx_ref = c * image_elements + x + kx + ros_offset.x() + (y + ky + ros_offset.y()) * dims.x();
            //out_matrix[idx_in*coefficients+coeff_counter++] = ref_data[idx_ref];
            out_matrix[idx_in+(coeff_counter++)*klocations] = ref_data[idx_ref];

          }
        }
      }

      //Loop over target coils here
      for (unsigned int c = 0; c < target_coils; c++) {
        //b[idx_in*coils + c] = ref_data[c*image_elements + y*dims.x+x];
/* DPCT_ORIG         b[idx_in + c*klocations] = ref_data[c*image_elements + (y+ros_offset.y)*dims.x+(x+ros_offset.x)];*/
        b[idx_in + c * klocations] =
            ref_data[c * image_elements + (y + ros_offset.y()) * dims.x() + (x + ros_offset.x())];
      }
    }
  }

  //TODO: This should take source and target coils into consideration
/* DPCT_ORIG   template <class T> __global__ void copy_grappa_coefficients_to_kernel_2d(const T* __restrict__ coeffs,
                                                                           T* __restrict__ kernel,
                                                                           int source_coils,
                                                                           int target_coils,
                                                                           int2 kernel_size,
                                                                           int acceleration_factor,
                                                                           int set)*/
  template <class T>
  void copy_grappa_coefficients_to_kernel_2d(const T* __restrict__ coeffs, T* __restrict__ kernel, int source_coils,
                                             int target_coils, sycl::int2 kernel_size, int acceleration_factor, int set)
  {
/* DPCT_ORIG     unsigned long idx_in = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned long idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

/* DPCT_ORIG     unsigned int coefficients_in_set = source_coils*kernel_size.x*kernel_size.y*target_coils;*/
    unsigned int coefficients_in_set = source_coils * kernel_size.x() * kernel_size.y() * target_coils;

    if (idx_in < coefficients_in_set) {
      int idx_in_tmp = idx_in;
/* DPCT_ORIG       int kx = idx_in%kernel_size.x;*/
      int kx = idx_in % kernel_size.x();
/* DPCT_ORIG       idx_in = (idx_in-kx)/kernel_size.x;*/
      idx_in = (idx_in - kx) / kernel_size.x();
/* DPCT_ORIG       int ky = idx_in%kernel_size.y;*/
      int ky = idx_in % kernel_size.y();
/* DPCT_ORIG       idx_in = (idx_in-ky)/kernel_size.y;*/
      idx_in = (idx_in - ky) / kernel_size.y();
      int coil = idx_in%source_coils;
      idx_in = (idx_in-coil)/source_coils;
      int coilg = idx_in;

/* DPCT_ORIG       kernel[coilg*source_coils*(kernel_size.y*acceleration_factor)*kernel_size.x +*/
      kernel[coilg * source_coils * (kernel_size.y() * acceleration_factor) * kernel_size.x() +
             /* DPCT_ORIG              coil*(kernel_size.y*acceleration_factor)*kernel_size.x +*/
             coil * (kernel_size.y() * acceleration_factor) * kernel_size.x() +
             /* DPCT_ORIG              (ky*acceleration_factor + set + 1)*kernel_size.x + kx] = coeffs[idx_in_tmp];*/
             (ky * acceleration_factor + set + 1) * kernel_size.x() + kx] = coeffs[idx_in_tmp];

      if ((coil == coilg) && (kx == 0) && (ky == 0) && (set == 0)) {
/* DPCT_ORIG         kernel[coilg*source_coils*(kernel_size.y*acceleration_factor)*kernel_size.x +*/
        kernel[coilg * source_coils * (kernel_size.y() * acceleration_factor) * kernel_size.x() +
               /* DPCT_ORIG                coil*(kernel_size.y*acceleration_factor)*kernel_size.x +*/
               coil * (kernel_size.y() * acceleration_factor) * kernel_size.x() +
               /* DPCT_ORIG                ((kernel_size.y>>1)*acceleration_factor)*kernel_size.x + (kernel_size.x>>1)
                  ]._real = 1;*/
               ((kernel_size.y() >> 1) * acceleration_factor) * kernel_size.x() +
               (kernel_size.x() >> 1)]
            ._real = 1;
      }
    }
  }

/* DPCT_ORIG   template <class T> __global__ void copy_grappa_kernel_to_kspace_2d(const T* __restrict__ kernel,
                                                                     T* __restrict__ out,
                                                                     int2 dims,
                                                                     int2 kernel_size,
                                                                     int coils)*/
  template <class T>
  void copy_grappa_kernel_to_kspace_2d(const T* __restrict__ kernel, T* __restrict__ out, sycl::int2 dims,
                                       sycl::int2 kernel_size, int coils)
  {

/* DPCT_ORIG     unsigned long idx_in = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned long idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

/* DPCT_ORIG     if (idx_in < kernel_size.x*kernel_size.y*coils) {*/
    if (idx_in < kernel_size.x() * kernel_size.y() * coils) {
      int idx_in_tmp = idx_in;
/* DPCT_ORIG       int kx = idx_in%kernel_size.x;*/
      int kx = idx_in % kernel_size.x();
/* DPCT_ORIG       idx_in = (idx_in-kx)/kernel_size.x;*/
      idx_in = (idx_in - kx) / kernel_size.x();
/* DPCT_ORIG       int ky = idx_in%kernel_size.y;*/
      int ky = idx_in % kernel_size.y();
/* DPCT_ORIG       idx_in = (idx_in-ky)/kernel_size.y;*/
      idx_in = (idx_in - ky) / kernel_size.y();
      int coil = idx_in;

/* DPCT_ORIG       int outx = -(kx- (kernel_size.x>>1)) + (dims.x>>1); */
      int outx = -(kx - (kernel_size.x() >> 1)) + (dims.x() >> 1); // Flipping the kernel for conv
/* DPCT_ORIG       int outy = -(ky- (kernel_size.y>>1)) + (dims.y>>1);*/
      int outy = -(ky - (kernel_size.y() >> 1)) + (dims.y() >> 1);

/* DPCT_ORIG       out[coil*dims.x*dims.y + outy*dims.x + outx] = kernel[idx_in_tmp];*/
      out[coil * dims.x() * dims.y() + outy * dims.x() + outx] = kernel[idx_in_tmp];
    }
  }

/* DPCT_ORIG   __global__ void scale_and_add_unmixing_coeffs(const complext<float> * __restrict__ unmixing,
                                                const complext<float> * __restrict__ csm,
                                                complext<float> * __restrict__ out,
                                                int elements,
                                                int coils,
                                                float scale_factor)*/
  void scale_and_add_unmixing_coeffs(const complext<float>* __restrict__ unmixing,
                                     const complext<float>* __restrict__ csm, complext<float>* __restrict__ out,
                                     int elements, int coils, float scale_factor)
  {
/* DPCT_ORIG     unsigned long idx_in = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned long idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    complext<float>  tmp;
    if (idx_in < elements) {
      for (int c = 0; c < coils; c++) {
        tmp = unmixing[c*elements + idx_in]*conj(csm[idx_in]);
        out[c*elements + idx_in] += scale_factor*tmp;

      }
    }
  }

/* DPCT_ORIG   __global__ void scale_and_copy_unmixing_coeffs(const complext<float> * __restrict__ unmixing,
                                                 complext<float> * __restrict__ out,
                                                 int elements,
                                                 int coils,
                                                 float scale_factor)*/
  void scale_and_copy_unmixing_coeffs(const complext<float>* __restrict__ unmixing, complext<float>* __restrict__ out,
                                      int elements, int coils, float scale_factor)
  {
/* DPCT_ORIG     unsigned long idx_in = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned long idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if (idx_in < elements) {
      for (int c = 0; c < coils; c++) {
        out[c*elements + idx_in] = scale_factor*unmixing[c*elements + idx_in];

      }
    }
  }

/* DPCT_ORIG   __global__ void conj_csm_coeffs(const complext<float> * __restrict__ csm,
                                  complext<float> * __restrict__ out,
                                  int source_elements,
                                  int target_elements)*/
  void conj_csm_coeffs(const complext<float>* __restrict__ csm, complext<float>* __restrict__ out, int source_elements,
                       int target_elements)
  {
    //TODO: Here we need to have both src_elements and target_elements and we use conj(csm) for all target_elements and 0.0 when element > target_elements

/* DPCT_ORIG     unsigned long idx_in = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned long idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if (idx_in < source_elements) {
      if (idx_in >= target_elements) {
        out[idx_in] = complext<float> (0.0,0.0);
      } else {
        out[idx_in] = conj(csm[idx_in]);
      }
    }
  }

/* DPCT_ORIG   __global__ void single_channel_coeffs(complext<float> * out,
                                        int channel_no,
                                        int elements_per_channel)*/
  void single_channel_coeffs(complext<float>* out, int channel_no, int elements_per_channel)
  {
/* DPCT_ORIG     unsigned long idx_in = blockIdx.x*blockDim.x+threadIdx.x;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    unsigned long idx_in = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if (idx_in < elements_per_channel) {
      out[idx_in + channel_no*elements_per_channel] = complext<float>(1.0,0.0);
    }
  }

  template <class T>
  int htgrappa_calculate_grappa_unmixing(cuNDArray<T>* ref_data, cuNDArray<T>* b1, unsigned int acceleration_factor,
                                         std::vector<unsigned int>* kernel_size, cuNDArray<T>* out_mixing_coeff,
                                         std::vector<std::pair<unsigned int, unsigned int>>* sampled_region,
                                         std::list<unsigned int>* uncombined_channels) try {

    if (ref_data->get_number_of_dimensions() != b1->get_number_of_dimensions()) {
      std::cerr << "htgrappa_calculate_grappa_unmixing: Dimensions mismatch" << std::endl;
      return -1;
    }

    for (unsigned int i = 0; i < (ref_data->get_number_of_dimensions()-1); i++) {
      if (ref_data->get_size(i) != b1->get_size(i)) {
        std::cerr << "htgrappa_calculate_grappa_unmixing: Dimensions mismatch" << std::endl;
        return -1;
      }
    }

    unsigned int RO = ref_data->get_size(0);
    unsigned int E1 = ref_data->get_size(1);
    unsigned int source_coils = ref_data->get_size(ref_data->get_number_of_dimensions()-1);
    unsigned int target_coils = b1->get_size(b1->get_number_of_dimensions()-1);
    unsigned int elements_per_coil = b1->get_number_of_elements()/target_coils;

    if (target_coils > source_coils) {
      std::cerr << "target_coils > source_coils" << std::endl;
      return -1;
    }

    if (acceleration_factor == 1) {
/* DPCT_ORIG       dim3 blockDim(512,1,1);*/
      dpct::dim3 blockDim(512, 1, 1);
/* DPCT_ORIG       dim3 gridDim((unsigned int) std::ceil((1.0f*elements_per_coil*source_coils)/blockDim.x), 1, 1 );*/
      dpct::dim3 gridDim((unsigned int)std::ceil((1.0f * elements_per_coil * source_coils) / blockDim.x), 1, 1);

/* DPCT_ORIG       conj_csm_coeffs<<< gridDim, blockDim >>>( b1->get_data_ptr(),
                                                out_mixing_coeff->get_data_ptr(),
                                                out_mixing_coeff->get_number_of_elements(),
                                                b1->get_number_of_elements());*/
      /*
      DPCT1049:55: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
      info::device::max_work_group_size. Adjust the work-group size if needed.
      */
        {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                auto b1_get_data_ptr_ct0 = b1->get_data_ptr();
                auto out_mixing_coeff_get_data_ptr_ct1 = out_mixing_coeff->get_data_ptr();
                auto out_mixing_coeff_get_number_of_elements_ct2 = out_mixing_coeff->get_number_of_elements();
                auto b1_get_number_of_elements_ct3 = b1->get_number_of_elements();

                cgh.parallel_for<dpct_kernel_name<class conj_csm_coeffs_8dba88>>(
                    sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                        conj_csm_coeffs(b1_get_data_ptr_ct0, out_mixing_coeff_get_data_ptr_ct1,
                                        out_mixing_coeff_get_number_of_elements_ct2, b1_get_number_of_elements_ct3);
                    });
            });
        }

      std::list<unsigned int>::iterator it;
/* DPCT_ORIG       gridDim = dim3((unsigned int) std::ceil((1.0f*(elements_per_coil))/blockDim.x), 1, 1 );*/
      gridDim = dpct::dim3((unsigned int)std::ceil((1.0f * (elements_per_coil)) / blockDim.x), 1, 1);
      int uncombined_channel_no = 0;
      if (uncombined_channels) {
        for ( it = uncombined_channels->begin(); it != uncombined_channels->end(); it++ ) {
          uncombined_channel_no++;
          //TODO: Adjust pointers to reflect that number of target/source may not be qual
/* DPCT_ORIG           single_channel_coeffs<<< gridDim, blockDim >>>( out_mixing_coeff->get_data_ptr() +
   uncombined_channel_no*source_coils*elements_per_coil, *it, (elements_per_coil));*/
          /*
          DPCT1049:56: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit,
          query info::device::max_work_group_size. Adjust the work-group size if needed.
          */
                {
                    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

                    dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                        auto out_mixing_coeff_get_data_ptr_uncombined_channel_no_source_coils_elements_per_coil_ct0 =
                            out_mixing_coeff->get_data_ptr() + uncombined_channel_no * source_coils * elements_per_coil;
                        auto it_ct1 = *it;

                        cgh.parallel_for<dpct_kernel_name<class single_channel_coeffs_ab1451>>(
                            sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                                single_channel_coeffs(
                                    out_mixing_coeff_get_data_ptr_uncombined_channel_no_source_coils_elements_per_coil_ct0,
                                    it_ct1, (elements_per_coil));
                            });
                    });
                }
        }
      }
      return 0;
    }

    if (kernel_size->size() != (ref_data->get_number_of_dimensions()-1)) {
      std::cerr << "htgrappa_calculate_grappa_unmixing: Kernel size does not match the data dimensions" << std::endl;
      return -1;
    }

    if (ref_data->get_number_of_dimensions() > 3) {
      std::cerr << "htgrappa_calculate_grappa_unmixing: Not yet implemented for 3D" << std::endl;
      return -1;
    }

    //Calculate region of support + offsets
    std::vector<size_t> rosTmp = ref_data->get_dimensions();

    std::vector<unsigned int> ros(rosTmp.size());
    for ( unsigned int ii=0; ii<rosTmp.size(); ii++ ){
      ros[ii] = rosTmp[ii];
    }

    ros.pop_back(); //Remove the number of coils
    std::vector<unsigned int> ros_offset(ref_data->get_number_of_dimensions(),0);
    unsigned long int kspace_locations = 1;

    if (sampled_region) {
      for (unsigned int i = 0; i < ros.size(); i++) {
        if (i > 0) {
          ros[i] = (*sampled_region)[i].second-(*sampled_region)[i].first-((*kernel_size)[i]*acceleration_factor);
        } else {
          ros[i] = (*sampled_region)[i].second-(*sampled_region)[i].first-(*kernel_size)[i];
        }
        ros_offset[i] = (*sampled_region)[i].first+(((*sampled_region)[i].second-(*sampled_region)[i].first-ros[i])>>1);
        kspace_locations *= ros[i];
      }
    } else {
      for (unsigned int i = 0; i < ros.size(); i++) {
        if (i > 0) {
          ros[i] -= ((*kernel_size)[i]*acceleration_factor);
        } else {
          ros[i] -= (*kernel_size)[i];
        }
        ros_offset[i] = (ref_data->get_size(i)-ros[i])>>1;
        kspace_locations *= ros[i];
      }
    }

    /*
      for (unsigned int i = 0; i < ros.size(); i++) {
      GDEBUG_STREAM("ROS[" << i << "] = " << ros[i] << " + " << ros_offset[i] << std::endl);
      }
    */

    std::vector<size_t> sys_matrix_size;
    sys_matrix_size.push_back(kspace_locations);
    sys_matrix_size.push_back(source_coils*(*kernel_size)[0]*(*kernel_size)[1]);

    std::vector<size_t> b_size;
    b_size.push_back(kspace_locations);
    b_size.push_back(target_coils);

    cuNDArray<T> system_matrix = cuNDArray<T>(sys_matrix_size);

    clear(&system_matrix);

    cuNDArray<T> b = cuNDArray<T>(b_size);

    std::vector<size_t> dimTmp = ref_data->get_dimensions();
    std::vector<unsigned int> dimInt(2, 0);
    dimInt[0] = dimTmp[0];
    dimInt[1] = dimTmp[1];

/* DPCT_ORIG     int2 dims = vec_to_int2(dimInt);*/
    sycl::int2 dims = vec_to_int2(dimInt);
/* DPCT_ORIG     int2 dros = vec_to_int2(ros);*/
    sycl::int2 dros = vec_to_int2(ros);
/* DPCT_ORIG     int2 dros_offset = vec_to_int2(ros_offset);*/
    sycl::int2 dros_offset = vec_to_int2(ros_offset);
/* DPCT_ORIG     int2 dkernel_size = vec_to_int2(*kernel_size);*/
    sycl::int2 dkernel_size = vec_to_int2(*kernel_size);

    //TODO: Use source coils here
    int n = source_coils*(*kernel_size)[0]*(*kernel_size)[1];
    int m = kspace_locations;

    std::vector<size_t> AHA_dims(2,n);
    cuNDArray<T> AHA = cuNDArray<T>(AHA_dims);
    cuNDArray<T> AHA_set0 = cuNDArray<T>(AHA_dims);

    hoNDArray<T> AHA_host(n, n);
/* DPCT_ORIG     float2* pAHA = (float2*) AHA_host.get_data_ptr();*/
    sycl::float2* pAHA = (sycl::float2*)AHA_host.get_data_ptr();

    //TODO: Use target coils here
    std::vector<size_t> AHrhs_dims;
    AHrhs_dims.push_back(n);
    AHrhs_dims.push_back(target_coils);

    cuNDArray<T> AHrhs = cuNDArray<T>(AHrhs_dims);

/* DPCT_ORIG     cublasHandle_t handle = *CUBLASContextProvider::instance()->getCublasHandle();*/
    dpct::blas::descriptor_ptr handle = *CUBLASContextProvider::instance()->getCublasHandle();

    std::vector<size_t> gkernel_dims;
    gkernel_dims.push_back((*kernel_size)[0]);
    gkernel_dims.push_back((*kernel_size)[1]*acceleration_factor);
    gkernel_dims.push_back(source_coils);
    gkernel_dims.push_back(target_coils);
    cuNDArray<T> gkernel = cuNDArray<T>(gkernel_dims);
    clear(&gkernel);

    //GadgetronTimer timer;

    for (unsigned int set = 0; set < acceleration_factor-1; set++)
      {
        //GDEBUG_STREAM("Calculating coefficients for set " << set << std::endl);

        //GDEBUG_STREAM("dros.x = " << dros.x << ", dros.y = " << dros.y << std::endl);

        std::ostringstream ostr;
        ostr << "Set_" << set << "_";
        std::string appendix = ostr.str();

/* DPCT_ORIG         dim3 blockDim(512,1,1);*/
        dpct::dim3 blockDim(512, 1, 1);
/* DPCT_ORIG         dim3 gridDim((unsigned int) std::ceil((1.0f*kspace_locations)/blockDim.x), 1, 1 );*/
        dpct::dim3 gridDim((unsigned int)std::ceil((1.0f * kspace_locations) / blockDim.x), 1, 1);

/* DPCT_ORIG         form_grappa_system_matrix_kernel_2d<<< gridDim, blockDim >>>( ref_data->get_data_ptr(), dims,
                                                                      source_coils, target_coils, dros, dros_offset,
                                                                      dkernel_size, acceleration_factor, set,
                                                                      system_matrix.get_data_ptr(),
                                                                      b.get_data_ptr());*/
        /*
        DPCT1049:57: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto ref_data_get_data_ptr_ct0 = ref_data->get_data_ptr();
            auto system_matrix_get_data_ptr_ct9 = system_matrix.get_data_ptr();
            auto b_get_data_ptr_ct10 = b.get_data_ptr();

            /*
            DPCT1050:296: The template argument of the dpct_kernel_name could not be deduced. You need to update this
            code.
            */
            cgh.parallel_for<dpct_kernel_name<class form_grappa_system_matrix_kernel_2d_77f215,
                                              dpct_placeholder /*Fix the type mannually*/>>(
                sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                    form_grappa_system_matrix_kernel_2d(ref_data_get_data_ptr_ct0, dims, source_coils, target_coils,
                                                        dros, dros_offset, dkernel_size, acceleration_factor, set,
                                                        system_matrix_get_data_ptr_ct9, b_get_data_ptr_ct10);
                });
        });

/* DPCT_ORIG         cudaError_t err = cudaGetLastError();*/
        /*
        DPCT1010:226: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError
        function call was replaced with 0. You need to rewrite this code.
        */
        dpct::err0 err = 0;
/* DPCT_ORIG         if( err != cudaSuccess ){*/
        /*
        DPCT1000:223: Error handling if-stmt was detected but could not be rewritten.
        */
        if (err != 0) {
          /*
          DPCT1001:222: The statement could not be removed.
          */
          std::cerr << "htgrappa_calculate_grappa_unmixing: Unable to form system matrix: " <<
              /* DPCT_ORIG             cudaGetErrorString(err) << std::endl;*/
              /*
              DPCT1009:227: SYCL reports errors using exceptions and does not use error codes. Please replace the
              "get_error_string_dummy(...)" with a real error-handling function.
              */
              dpct::get_error_string_dummy(err)
                    << std::endl;
          return -1;
        }

        //  {
        //      std::string filename = debugFolder+appendix+"A.cplx";
            //write_cuNDArray_to_disk(&system_matrix, filename.c_str());
        //  }

        //  {
        //      std::string filename = debugFolder+appendix+"b.cplx";
            //write_cuNDArray_to_disk(&b, filename.c_str());
        //  }

        complext<float>  alpha = complext<float>(1);
        complext<float>  beta = complext<float>(0);

/* DPCT_ORIG         cublasStatus_t stat;*/
        int stat;

        if ( set == 0 )
          {
            {
              //GPUTimer t2("Cgemm call");
/* DPCT_ORIG               stat = cublasCgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N,
                                 n,n,m,(float2*) &alpha,
                                 (float2*) system_matrix.get_data_ptr(), m,
                                 (float2*) system_matrix.get_data_ptr(), m,
                                 (float2*) &beta, (float2*) AHA.get_data_ptr(), n);*/
              stat = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
                  handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::nontrans, n, n, m,
                  dpct::get_value((sycl::float2*)&alpha, handle->get_queue()),
                  (std::complex<float>*)system_matrix.get_data_ptr(), m,
                  (std::complex<float>*)system_matrix.get_data_ptr(), m,
                  dpct::get_value((sycl::float2*)&beta, handle->get_queue()), (std::complex<float>*)AHA.get_data_ptr(),
                  n));

/* DPCT_ORIG               if (stat != CUBLAS_STATUS_SUCCESS) {*/
              if (stat != 0) {
                std::cerr << "htgrappa_calculate_grappa_unmixing: Failed to form AHA product using cublas gemm" << std::endl;
                std::cerr << "---- cublas error code " << stat << std::endl;
                return -1;
              }
            }

            {
              //timer.start("copy AHA to host");
/* DPCT_ORIG               if (cudaMemcpy(pAHA, AHA.get_data_ptr(), AHA_host.get_number_of_bytes(),
 * cudaMemcpyDeviceToHost) != cudaSuccess)*/
              if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                       .memcpy(pAHA, AHA.get_data_ptr(), AHA_host.get_number_of_bytes())
                                       .wait()) != 0)
                {
                  std::cerr << "htgrappa_calculate_grappa_unmixing: Failed to copy AHA to host" << std::endl;
                  std::cerr << "---- cublas error code " << stat << std::endl;
                  return -1;
                }
              //timer.stop();

              //timer.start("apply the regularization");
              // apply the regularization
              double lamda = 0.0005;

/* DPCT_ORIG               double trA = std::sqrt(pAHA[0].x*pAHA[0].x + pAHA[0].y*pAHA[0].y);*/
              double trA = std::sqrt(pAHA[0].x() * pAHA[0].x() + pAHA[0].y() * pAHA[0].y());
              size_t c;
              for ( c=1; c<n; c++ )
                {
/* DPCT_ORIG                   float x = pAHA[c+c*n].x;*/
                  float x = pAHA[c + c * n].x();
/* DPCT_ORIG                   float y = pAHA[c+c*n].y;*/
                  float y = pAHA[c + c * n].y();
                  trA += std::sqrt(x*x+y*y);
                }

              double value = trA*lamda/n;
              for ( c=0; c<n; c++ )
                {
/* DPCT_ORIG                   float x = pAHA[c+c*n].x;*/
                  float x = pAHA[c + c * n].x();
/* DPCT_ORIG                   float y = pAHA[c+c*n].y;*/
                  float y = pAHA[c + c * n].y();
/* DPCT_ORIG                   pAHA[c+c*n].x = std::sqrt(x*x+y*y) + value;*/
                  pAHA[c + c * n].x() = std::sqrt(x * x + y * y) + value;
/* DPCT_ORIG                   pAHA[c+c*n].y = 0;*/
                  pAHA[c + c * n].y() = 0;
                }
              //timer.stop();

              //timer.start("copy the AHA to device");
/* DPCT_ORIG               if (cudaMemcpy(AHA.get_data_ptr(), pAHA, AHA_host.get_number_of_bytes(),
 * cudaMemcpyHostToDevice) != cudaSuccess)*/
              if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                       .memcpy(AHA.get_data_ptr(), pAHA, AHA_host.get_number_of_bytes())
                                       .wait()) != 0)
                {
                  std::cerr << "htgrappa_calculate_grappa_unmixing: Failed to copy regularized AHA to device" << std::endl;
                  std::cerr << "---- cublas error code " << stat << std::endl;
                  return -1;
                }
              //timer.stop();
            }

            AHA_set0 = AHA;
          }
        else
          {
            AHA = AHA_set0;
          }

        //  {
        //      std::string filename = debugFolder+appendix+"AHA.cplx";
        //write_cuNDArray_to_disk(&AHA, filename.c_str());
        //  }

        {

          //GPUTimer timer("GRAPPA cublas gemm");
          //TODO: Sort out arguments for source and target coils here.
/* DPCT_ORIG           stat = cublasCgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N,
                             n,target_coils,m,(float2*) &alpha,
                             (float2*) system_matrix.get_data_ptr(), m,
                             (float2*) b.get_data_ptr(), m,
                             (float2*) &beta, (float2*)AHrhs.get_data_ptr(), n);*/
          stat = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
              handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::nontrans, n, target_coils,
              m, dpct::get_value((sycl::float2*)&alpha, handle->get_queue()),
              (std::complex<float>*)system_matrix.get_data_ptr(), m, (std::complex<float>*)b.get_data_ptr(), m,
              dpct::get_value((sycl::float2*)&beta, handle->get_queue()), (std::complex<float>*)AHrhs.get_data_ptr(),
              n));
        }

        //  {
        //      std::string filename = debugFolder+appendix+"AHrhs.cplx";
        //write_cuNDArray_to_disk(&AHrhs, filename.c_str());
        //  }

/* DPCT_ORIG         if (stat != CUBLAS_STATUS_SUCCESS) {*/
        if (stat != 0) {
          std::cerr << "htgrappa_calculate_grappa_unmixing: Failed to form AHrhs product using cublas gemm" << std::endl;
          std::cerr << "---- cublas error code " << stat << std::endl;
          return -1;
        }




    {
      //It actually turns out to be faster to do this inversion on the CPU. Problem is probably too small for GPU to make sense
      //GPUTimer cpu_invert_time("CPU Inversion time");


      ht_grappa_solve_spd_system(AHA, AHrhs);

    }

/* DPCT_ORIG         gridDim = dim3((unsigned int) std::ceil((1.0f*n*source_coils)/blockDim.x), 1, 1 );*/
        gridDim = dpct::dim3((unsigned int)std::ceil((1.0f * n * source_coils) / blockDim.x), 1, 1);

        //TODO: This should be target coils used as argument here.
/* DPCT_ORIG         copy_grappa_coefficients_to_kernel_2d<<< gridDim, blockDim >>>( AHrhs.get_data_ptr(),
                                                                        gkernel.get_data_ptr(),
                                                                        source_coils,
                                                                        target_coils,
                                                                        dkernel_size,
                                                                        acceleration_factor,
                                                                        set);*/
        /*
        DPCT1049:58: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto AHrhs_get_data_ptr_ct0 = AHrhs.get_data_ptr();
            auto gkernel_get_data_ptr_ct1 = gkernel.get_data_ptr();

            /*
            DPCT1050:297: The template argument of the dpct_kernel_name could not be deduced. You need to update this
            code.
            */
            cgh.parallel_for<dpct_kernel_name<class copy_grappa_coefficients_to_kernel_2d_623b46,
                                              dpct_placeholder /*Fix the type mannually*/>>(
                sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                    copy_grappa_coefficients_to_kernel_2d(AHrhs_get_data_ptr_ct0, gkernel_get_data_ptr_ct1,
                                                          source_coils, target_coils, dkernel_size, acceleration_factor,
                                                          set);
                });
        });

        //  {
        //      std::string filename = debugFolder+appendix+"kernel.cplx";
            //write_cuNDArray_to_disk(&gkernel, filename.c_str());
        //  }

/* DPCT_ORIG         err = cudaGetLastError();*/
        /*
        DPCT1010:228: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError
        function call was replaced with 0. You need to rewrite this code.
        */
        err = 0;
/* DPCT_ORIG         if( err != cudaSuccess ){*/
        /*
        DPCT1000:225: Error handling if-stmt was detected but could not be rewritten.
        */
        if (err != 0) {
          /*
          DPCT1001:224: The statement could not be removed.
          */
          std::cerr << "htgrappa_calculate_grappa_unmixing: Failed to copy calculated coefficients to kernel: " <<
              /* DPCT_ORIG             cudaGetErrorString(err) << std::endl;*/
              /*
              DPCT1009:229: SYCL reports errors using exceptions and does not use error codes. Please replace the
              "get_error_string_dummy(...)" with a real error-handling function.
              */
              dpct::get_error_string_dummy(err)
                    << std::endl;
          return -1;
        }

      }

    //{
    //    std::string filename = debugFolder+"kernel_all.cplx";
    //    write_cuNDArray_to_disk(&gkernel, filename.c_str());
    //}

    //TODO: This should be source coils
    cuNDArray<T> tmp_mixing = cuNDArray<T>(ref_data->get_dimensions());

    int kernel_elements = gkernel.get_number_of_elements()/target_coils;
    int total_elements = tmp_mixing.get_number_of_elements()/source_coils;
/* DPCT_ORIG     dkernel_size.y *= acceleration_factor;*/
    dkernel_size.y() *= acceleration_factor;

    std::vector<size_t> ft_dims(2,0);ft_dims[1] = 1;
    clear(out_mixing_coeff);
    unsigned int current_uncombined_index = 0;

    //TODO: Loop over target coils.
    for (unsigned int c = 0; c < target_coils; c++)
      {
        clear(&tmp_mixing);

/* DPCT_ORIG         dim3 blockDim(512,1,1);*/
        dpct::dim3 blockDim(512, 1, 1);
/* DPCT_ORIG         dim3 gridDim((unsigned int) std::ceil((1.0f*kernel_elements)/blockDim.x), 1, 1 );*/
        dpct::dim3 gridDim((unsigned int)std::ceil((1.0f * kernel_elements) / blockDim.x), 1, 1);

        //TODO: Take source and target into consideration
/* DPCT_ORIG         copy_grappa_kernel_to_kspace_2d<<< gridDim, blockDim
   >>>((gkernel.get_data_ptr()+(c*kernel_elements)), tmp_mixing.get_data_ptr(), dims, dkernel_size, source_coils);*/
        /*
        DPCT1049:59: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
            auto gkernel_get_data_ptr_c_kernel_elements_ct0 = (gkernel.get_data_ptr() + (c * kernel_elements));
            auto tmp_mixing_get_data_ptr_ct1 = tmp_mixing.get_data_ptr();

            /*
            DPCT1050:298: The template argument of the dpct_kernel_name could not be deduced. You need to update this
            code.
            */
            cgh.parallel_for<dpct_kernel_name<class copy_grappa_kernel_to_kspace_2d_e20147,
                                              dpct_placeholder /*Fix the type mannually*/>>(
                sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                    copy_grappa_kernel_to_kspace_2d(gkernel_get_data_ptr_c_kernel_elements_ct0,
                                                    tmp_mixing_get_data_ptr_ct1, dims, dkernel_size, source_coils);
                });
        });

/* DPCT_ORIG         cudaError_t err = cudaGetLastError();*/
        /*
        DPCT1010:234: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError
        function call was replaced with 0. You need to rewrite this code.
        */
        dpct::err0 err = 0;
/* DPCT_ORIG         if( err != cudaSuccess ){*/
        /*
        DPCT1000:231: Error handling if-stmt was detected but could not be rewritten.
        */
        if (err != 0) {
          /*
          DPCT1001:230: The statement could not be removed.
          */
          std::cerr << "htgrappa_calculate_grappa_unmixing: Unable to pad GRAPPA kernel: " <<
              /* DPCT_ORIG             cudaGetErrorString(err) << std::endl;*/
              /*
              DPCT1009:235: SYCL reports errors using exceptions and does not use error codes. Please replace the
              "get_error_string_dummy(...)" with a real error-handling function.
              */
              dpct::get_error_string_dummy(err)
                    << std::endl;
          return -1;
        }

        cuNDFFT<typename realType<T>::Type>::instance()->ifft(&tmp_mixing, &ft_dims);

        float scale_factor = (float)std::sqrt((double)(RO*E1));

/* DPCT_ORIG         gridDim = dim3((unsigned int) std::ceil(1.0f*total_elements/blockDim.x), 1, 1 );*/
        gridDim = dpct::dim3((unsigned int)std::ceil(1.0f * total_elements / blockDim.x), 1, 1);
/* DPCT_ORIG         scale_and_add_unmixing_coeffs<<< gridDim, blockDim >>>(tmp_mixing.get_data_ptr(),
                                                               (b1->get_data_ptr()+ c*total_elements),
                                                               out_mixing_coeff->get_data_ptr(),
                                                               total_elements,
                                                               source_coils,
                                                               scale_factor);*/
        /*
        DPCT1049:60: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                auto tmp_mixing_get_data_ptr_ct0 = tmp_mixing.get_data_ptr();
                auto b1_get_data_ptr_c_total_elements_ct1 = (b1->get_data_ptr() + c * total_elements);
                auto out_mixing_coeff_get_data_ptr_ct2 = out_mixing_coeff->get_data_ptr();

                cgh.parallel_for<dpct_kernel_name<class scale_and_add_unmixing_coeffs_ae2ed4>>(
                    sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                        scale_and_add_unmixing_coeffs(tmp_mixing_get_data_ptr_ct0, b1_get_data_ptr_c_total_elements_ct1,
                                                      out_mixing_coeff_get_data_ptr_ct2, total_elements, source_coils,
                                                      scale_factor);
                    });
            });
        }
/* DPCT_ORIG         err = cudaGetLastError();*/
        /*
        DPCT1010:236: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError
        function call was replaced with 0. You need to rewrite this code.
        */
        err = 0;
/* DPCT_ORIG         if( err != cudaSuccess ){*/
        /*
        DPCT1000:233: Error handling if-stmt was detected but could not be rewritten.
        */
        if (err != 0) {
          /*
          DPCT1001:232: The statement could not be removed.
          */
          std::cerr << "htgrappa_calculate_grappa_unmixing: scale and add mixing coeffs: " <<
              /* DPCT_ORIG             cudaGetErrorString(err) << std::endl;*/
              /*
              DPCT1009:237: SYCL reports errors using exceptions and does not use error codes. Please replace the
              "get_error_string_dummy(...)" with a real error-handling function.
              */
              dpct::get_error_string_dummy(err)
                    << std::endl;
          return -1;
        }

        if (uncombined_channels) {
          std::list<unsigned int>::iterator it = std::find((*uncombined_channels).begin(),(*uncombined_channels).end(),c);
          if (it != (*uncombined_channels).end()) {
            current_uncombined_index++;
/* DPCT_ORIG             scale_and_copy_unmixing_coeffs<<< gridDim, blockDim >>>(tmp_mixing.get_data_ptr(),
                                                                    (out_mixing_coeff->get_data_ptr()+current_uncombined_index*total_elements*source_coils),
                                                                    total_elements,
                                                                    source_coils,
                                                                    scale_factor);*/
            /*
            DPCT1049:61: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit,
            query info::device::max_work_group_size. Adjust the work-group size if needed.
            */
                dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                    auto tmp_mixing_get_data_ptr_ct0 = tmp_mixing.get_data_ptr();
                    auto out_mixing_coeff_get_data_ptr_current_uncombined_index_total_elements_source_coils_ct1 =
                        (out_mixing_coeff->get_data_ptr() + current_uncombined_index * total_elements * source_coils);

                    cgh.parallel_for<dpct_kernel_name<class scale_and_copy_unmixing_coeffs_2c0fe3>>(
                        sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                            scale_and_copy_unmixing_coeffs(
                                tmp_mixing_get_data_ptr_ct0,
                                out_mixing_coeff_get_data_ptr_current_uncombined_index_total_elements_source_coils_ct1,
                                total_elements, source_coils, scale_factor);
                        });
                });
          }
        }

      }

    //GDEBUG_STREAM("**********cublasDestroy()**************" << std::endl);
    //cublasDestroy_v2(handle);

    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <class T> int inverse_clib_matrix(cuNDArray<T>* A, cuNDArray<T>* b, cuNDArray<T>* coeff, double lamda) try {
    // A: M*N
    // b: M*K
    size_t M = A->get_size(0);
    size_t N = A->get_size(1);

    size_t K = b->get_size(1);

    std::vector<size_t> AHA_dims(2,N);
    cuNDArray<T> AHA = cuNDArray<T>(AHA_dims);

    std::vector<size_t> AHrhs_dims;
    AHrhs_dims.push_back(N);
    AHrhs_dims.push_back(K);

    coeff->create(AHrhs_dims);

/* DPCT_ORIG     cublasHandle_t handle = *CUBLASContextProvider::instance()->getCublasHandle();*/
    dpct::blas::descriptor_ptr handle = *CUBLASContextProvider::instance()->getCublasHandle();

    complext<float>  alpha = complext<float>(1);
    complext<float>  beta = complext<float>(0);

    //{
    //    std::string filename = debugFolder+"A.cplx";
    //    write_cuNDArray_to_disk(A, filename.c_str());
    //}

    //{
    //    std::string filename = debugFolder+"b.cplx";
    //    write_cuNDArray_to_disk(b, filename.c_str());
    //}

    {
      //GPUTimer t2("compute AHA ...");
/* DPCT_ORIG       cublasStatus_t stat = cublasCgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N,
                                        N,N,M,(float2*) &alpha,
                                        (float2*) A->get_data_ptr(), M,
                                        (float2*) A->get_data_ptr(), M,
                                        (float2*) &beta, (float2*) AHA.get_data_ptr(), N);*/
      int stat = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
          handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::nontrans, N, N, M,
          dpct::get_value((sycl::float2*)&alpha, handle->get_queue()), (std::complex<float>*)A->get_data_ptr(), M,
          (std::complex<float>*)A->get_data_ptr(), M, dpct::get_value((sycl::float2*)&beta, handle->get_queue()),
          (std::complex<float>*)AHA.get_data_ptr(), N));

/* DPCT_ORIG       if (stat != CUBLAS_STATUS_SUCCESS)*/
      if (stat != 0)
        {
          std::cerr << "inverse_clib_matrix: Failed to form AHA product using cublas gemm" << std::endl;
          std::cerr << "---- cublas error code " << stat << std::endl;
          return -1;
        }
    }

    //{
    //    std::string filename = debugFolder+"AHA.cplx";
    //    write_cuNDArray_to_disk(&AHA, filename.c_str());
    //}

    {
      //GPUTimer t2("compute AHrhs ...");
/* DPCT_ORIG       cublasStatus_t stat = cublasCgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N,
                                        N,K,M,(float2*) &alpha,
                                        (float2*) A->get_data_ptr(), M,
                                        (float2*) b->get_data_ptr(), M,
                                        (float2*) &beta, (float2*)coeff->get_data_ptr(), N);*/
      int stat = DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
          handle->get_queue(), oneapi::mkl::transpose::conjtrans, oneapi::mkl::transpose::nontrans, N, K, M,
          dpct::get_value((sycl::float2*)&alpha, handle->get_queue()), (std::complex<float>*)A->get_data_ptr(), M,
          (std::complex<float>*)b->get_data_ptr(), M, dpct::get_value((sycl::float2*)&beta, handle->get_queue()),
          (std::complex<float>*)coeff->get_data_ptr(), N));

/* DPCT_ORIG       if (stat != CUBLAS_STATUS_SUCCESS)*/
      if (stat != 0)
        {
          std::cerr << "inverse_clib_matrix: Failed to form AHrhs product using cublas gemm" << std::endl;
          std::cerr << "---- cublas error code " << stat << std::endl;
          return -1;
        }
    }

    //{
    //    std::string filename = debugFolder+"AHrhs.cplx";
    //    write_cuNDArray_to_disk(coeff, filename.c_str());
    //}

    // apply the regularization
    if ( lamda > 0 )
      {
        hoNDArray<T> AHA_host(N, N);
/* DPCT_ORIG         float2* pAHA = (float2*) AHA_host.get_data_ptr();*/
        sycl::float2* pAHA = (sycl::float2*)AHA_host.get_data_ptr();

        //GadgetronTimer timer;

        //timer.start("copy AHA to host");
/* DPCT_ORIG         if (cudaMemcpy(pAHA, AHA.get_data_ptr(), AHA_host.get_number_of_bytes(), cudaMemcpyDeviceToHost) !=
 * cudaSuccess)*/
        if (DPCT_CHECK_ERROR(
                dpct::get_in_order_queue().memcpy(pAHA, AHA.get_data_ptr(), AHA_host.get_number_of_bytes()).wait()) !=
            0)
          {
            std::cerr << "inverse_clib_matrix: Failed to copy AHA to host" << std::endl;
            return -1;
          }
        //timer.stop();

        //timer.start("apply the regularization");
        // apply the regularization
/* DPCT_ORIG         double trA = std::sqrt(pAHA[0].x*pAHA[0].x + pAHA[0].y*pAHA[0].y);*/
        double trA = std::sqrt(pAHA[0].x() * pAHA[0].x() + pAHA[0].y() * pAHA[0].y());
        size_t c;
        for ( c=1; c<N; c++ )
          {
/* DPCT_ORIG             float x = pAHA[c+c*N].x;*/
            float x = pAHA[c + c * N].x();
/* DPCT_ORIG             float y = pAHA[c+c*N].y;*/
            float y = pAHA[c + c * N].y();
            trA += std::sqrt(x*x+y*y);
          }

        double value = trA*lamda/N;
        for ( c=0; c<N; c++ )
          {
/* DPCT_ORIG             float x = pAHA[c+c*N].x;*/
            float x = pAHA[c + c * N].x();
/* DPCT_ORIG             float y = pAHA[c+c*N].y;*/
            float y = pAHA[c + c * N].y();
/* DPCT_ORIG             pAHA[c+c*N].x = std::sqrt(x*x+y*y) + value;*/
            pAHA[c + c * N].x() = std::sqrt(x * x + y * y) + value;
/* DPCT_ORIG             pAHA[c+c*N].y = 0;*/
            pAHA[c + c * N].y() = 0;
          }
        //timer.stop();

        //timer.start("copy the AHA to device");
/* DPCT_ORIG         if (cudaMemcpy(AHA.get_data_ptr(), pAHA, AHA_host.get_number_of_bytes(), cudaMemcpyHostToDevice) !=
 * cudaSuccess)*/
        if (DPCT_CHECK_ERROR(
                dpct::get_in_order_queue().memcpy(AHA.get_data_ptr(), pAHA, AHA_host.get_number_of_bytes()).wait()) !=
            0)
          {
            std::cerr << "inverse_clib_matrix: Failed to copy regularized AHA to device" << std::endl;
            return -1;
          }
        //timer.stop();
      }

    /*
      culaStatus s;
      s = culaDeviceCgels( 'N', N, N, K,
      (culaDeviceFloatComplex*)AHA.get_data_ptr(), N,
      (culaDeviceFloatComplex*)coeff->get_data_ptr(), N);
    */
    {
      //It actually turns out to be faster to do this inversion on the CPU. Problem is probably too small for GPU to make sense
      //GPUTimer cpu_invert_time("CPU Inversion time");

      ht_grappa_solve_spd_system(AHA, *coeff);

    }


    //{
    //    std::string filename = debugFolder+"coeff.cplx";
    //    write_cuNDArray_to_disk(coeff, filename.c_str());
    //}

    /*
    if (s != culaNoError)
      {
        GDEBUG_STREAM("inverse_clib_matrix: linear solve failed" << std::endl);
        return -1;
      }
    */
    return 0;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  //Template instanciation
  template EXPORTGPUPMRI int htgrappa_calculate_grappa_unmixing(cuNDArray<complext<float> >* ref_data,
                                                                cuNDArray<complext<float> >* b1,
                                                                unsigned int acceleration_factor,
                                                                std::vector<unsigned int> *kernel_size,
                                                                cuNDArray<complext<float> >* out_mixing_coeff,
                                                                std::vector< std::pair<unsigned int, unsigned int> >* sampled_region,
                                                                std::list< unsigned int >* uncombined_channels);

  template EXPORTGPUPMRI int inverse_clib_matrix(cuNDArray<complext<float> >* A,
                                                 cuNDArray<complext<float> >* b,
                                                 cuNDArray<complext<float> >* coeff,
                                                 double lamda);
}
