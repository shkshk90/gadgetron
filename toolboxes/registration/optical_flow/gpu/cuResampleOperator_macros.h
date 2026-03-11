#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_utils.h"
#include "complext.h"

/* 
   This macro definition is a workaround 
   for missing pure virtual device function support in Cuda.
   
   We provide this macro to avoid explicitly duplicating 
   the code below in every "cuResampleOperator-inherited" class.
*/

/*
DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1129:3: The type "vector_td<unsigned int, D>" is used in the SYCL kernel, but it is not device copyable. The
sycl::is_device_copyable specialization has been added for this type. Please review the code.
*/
#define DECLARE_CU_RESAMPLE_OPERATOR_SUPPORT(COMPONENT)                                                                \
                                                                                                                       \
 template <class T, unsigned int D>                                                                                    \
 void mult_M_kernel_batch(T* in, T* out, typename realType<T>::Type* displacements,                                    \
                          typename uintd<D>::Type matrix_size, unsigned int num_batches)                               \
 {                                                                                                                     \
  typedef typename realType<T>::Type REAL;                                                                             \
  const unsigned int idx = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(1) *                          \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_range(2) *                \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) +                \
                           sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *                          \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) +                \
                           sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);                        \
  const unsigned int num_elements = prod(matrix_size);                                                                 \
                                                                                                                       \
  if (idx < num_elements * num_batches) {                                                                              \
                                                                                                                       \
   const unsigned int batch_no = idx / num_elements;                                                                   \
   const unsigned int idx_in_batch = idx - batch_no * num_elements;                                                    \
   const typename uintd<D>::Type co = idx_to_co(idx_in_batch, matrix_size);                                            \
                                                                                                                       \
   typename reald<REAL, D>::Type co_disp = vector_td<REAL, D>(co);                                                     \
   for (unsigned int dim = 0; dim < D; dim++)                                                                          \
    co_disp.vec[dim] += displacements[dim * num_elements + idx_in_batch];                                              \
                                                                                                                       \
   out[idx] = interpolate<T, D>(batch_no, co_disp, matrix_size, in);                                                   \
  }                                                                                                                    \
 }                                                                                                                     \
                                                                                                                       \
 template <class T, unsigned int D>                                                                                    \
 void mult_M_kernel_extended(T* in, T* out, typename realType<T>::Type* displacements,                                 \
                             typename uintd<D>::Type matrix_size, unsigned int num_elements_in,                        \
                             unsigned int extended_size)                                                               \
 {                                                                                                                     \
  typedef typename realType<T>::Type REAL;                                                                             \
  const unsigned int idx = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(1) *                          \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_range(2) *                \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) +                \
                           sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *                          \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) +                \
                           sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);                        \
  const unsigned int num_elements_mat = prod(matrix_size);                                                             \
  const unsigned int num_elements_ext = prod(matrix_size) * extended_size;                                             \
                                                                                                                       \
  if (idx < num_elements_ext) {                                                                                        \
                                                                                                                       \
   const unsigned int batch_no = idx / num_elements_mat;                                                               \
   const unsigned int idx_in_batch = idx - batch_no * num_elements_mat;                                                \
                                                                                                                       \
   const typename uintd<D>::Type co = idx_to_co(idx_in_batch, matrix_size);                                            \
                                                                                                                       \
   typename reald<REAL, D>::Type co_disp = vector_td<REAL, D>(co);                                                     \
   for (unsigned int dim = 0; dim < D; dim++)                                                                          \
    co_disp.vec[dim] += displacements[dim * num_elements_ext + batch_no * num_elements_mat + idx_in_batch];            \
                                                                                                                       \
   out[idx] = interpolate<T, D>((idx >= num_elements_in) ? 0 : batch_no, co_disp, matrix_size, in);                    \
  }                                                                                                                    \
 }                                                                                                                     \
                                                                                                                       \
 template <class T, unsigned int D>                                                                                    \
 void cu##COMPONENT<T, D>::mult_M(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate)                                \
 {                                                                                                                     \
  if (!in || !out) {                                                                                                   \
   throw cuda_error("cuResampleOperator::mult_M(): illegal input/output array.");                                      \
  }                                                                                                                    \
                                                                                                                       \
  if (!this->offsets_.get()) {                                                                                         \
   throw cuda_error("cuResampleOperator::mult_M(): displacement field not set.");                                      \
  }                                                                                                                    \
                                                                                                                       \
  cuNDArray<T> tmp;                                                                                                    \
  if (accumulate) {                                                                                                    \
   tmp = *out;                                                                                                         \
  }                                                                                                                    \
                                                                                                                       \
  unsigned int num_disp_vectors = this->get_number_of_displacement_vectors();                                          \
  int surplus = this->offsets_->get_number_of_dimensions() - D;                                                        \
                                                                                                                       \
  if (!(surplus == 1 || surplus == 2) || this->offsets_->get_size(D - 1 + surplus) < D) {                              \
   throw cuda_error("cuResampleOperator::mult_M(): unexpected dimensions of displacement field.");                     \
  }                                                                                                                    \
                                                                                                                       \
  if (surplus == 1) {                                                                                                  \
   if (in->get_number_of_elements() != out->get_number_of_elements()) {                                                \
    throw cuda_error("cuResampleOperator::mult_M(): in/out array dimensions mismatch (1).");                           \
   }                                                                                                                   \
   if ((in->get_number_of_elements() % num_disp_vectors) != 0) {                                                       \
    throw cuda_error("cuResampleOperator::mult_M(): in/out array dimensions mismatch displacement field.");            \
   }                                                                                                                   \
  }                                                                                                                    \
                                                                                                                       \
  if (surplus == 2) {                                                                                                  \
   if ((out->get_number_of_elements() % in->get_number_of_elements()) != 0) {                                          \
    throw cuda_error("cuResampleOperator::mult_M(): in/out array dimensions mismatch (2).");                           \
   }                                                                                                                   \
   if (out->get_number_of_dimensions() != (D + 1) || out->get_number_of_elements() != num_disp_vectors) {              \
    throw cuda_error("cuResampleOperator::mult_M(): output array dimensions mismatch displacement field.");            \
   }                                                                                                                   \
  }                                                                                                                    \
                                                                                                                       \
  typename uint64d<D>::Type matrix_size = from_std_vector<size_t, D>(in->get_dimensions());                            \
  unsigned int num_elements_mat = prod(matrix_size);                                                                   \
  unsigned int num_batches = (surplus == 2) ? 1 : in->get_number_of_elements() / num_elements_mat;                     \
  unsigned int extended_dim = (surplus == 1) ? 1 : out->get_size(D);                                                   \
                                                                                                                       \
  dpct::dim3 blockDim, gridDim;                                                                                        \
                                                                                                                       \
  if (surplus == 1) {                                                                                                  \
   setup_grid(num_elements_mat, &blockDim, &gridDim, num_batches);                                                     \
  } else {                                                                                                             \
   setup_grid(num_elements_mat * extended_dim, &blockDim, &gridDim);                                                   \
  }                                                                                                                    \
                                                                                                                       \
  if (surplus == 1) {                                                                                                  \
   auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};       \
   dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});                        \
                                                                                                                       \
   dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {                                                         \
    auto in_get_data_ptr_ct0 = in->get_data_ptr();                                                                     \
    auto out_get_data_ptr_ct1 = out->get_data_ptr();                                                                   \
    auto this_offsets__get_data_ptr_ct2 = this->offsets_->get_data_ptr();                                              \
    auto vector_td_unsigned_int_D_matrix_size_ct3 = vector_td<unsigned int, D>(matrix_size);                           \
    auto num_batches_ct4 = num_batches;                                                                                \
                                                                                                                       \
    cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());                                      \
                                                                                                                       \
    cgh.parallel_for<dpct_kernel_name<class mult_M_kernel_batch_ffb871, T, dpct_kernel_scalar<D>>>(                    \
        sycl::nd_range<3>(gridDim * blockDim, blockDim), exp_props, [=](sycl::nd_item<3> item_ct1) {                   \
         mult_M_kernel_batch<T, D>(in_get_data_ptr_ct0, out_get_data_ptr_ct1, this_offsets__get_data_ptr_ct2,          \
                                   vector_td_unsigned_int_D_matrix_size_ct3, num_batches_ct4);                         \
        });                                                                                                            \
   });                                                                                                                 \
  } else {                                                                                                             \
   auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};       \
   dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});                        \
                                                                                                                       \
   dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {                                                         \
    auto in_get_data_ptr_ct0 = in->get_data_ptr();                                                                     \
    auto out_get_data_ptr_ct1 = out->get_data_ptr();                                                                   \
    auto this_offsets__get_data_ptr_ct2 = this->offsets_->get_data_ptr();                                              \
    auto vector_td_unsigned_int_D_matrix_size_ct3 = vector_td<unsigned int, D>(matrix_size);                           \
    auto in_get_number_of_elements_ct4 = in->get_number_of_elements();                                                 \
    auto extended_dim_ct5 = extended_dim;                                                                              \
                                                                                                                       \
    cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());                                      \
                                                                                                                       \
    cgh.parallel_for<dpct_kernel_name<class mult_M_kernel_extended_d16cbe, T, dpct_kernel_scalar<D>>>(                 \
        sycl::nd_range<3>(gridDim * blockDim, blockDim), exp_props, [=](sycl::nd_item<3> item_ct1) {                   \
         mult_M_kernel_extended<T, D>(in_get_data_ptr_ct0, out_get_data_ptr_ct1, this_offsets__get_data_ptr_ct2,       \
                                      vector_td_unsigned_int_D_matrix_size_ct3, in_get_number_of_elements_ct4,         \
                                      extended_dim_ct5);                                                               \
        });                                                                                                            \
   });                                                                                                                 \
  }                                                                                                                    \
                                                                                                                       \
  CHECK_FOR_CUDA_ERROR();                                                                                              \
                                                                                                                       \
  if (accumulate) {                                                                                                    \
   *out += tmp;                                                                                                        \
  }                                                                                                                    \
 }                                                                                                                     \
                                                                                                                       \
 template <class T, unsigned int D>                                                                                    \
 void mult_MH_kernel(T* in, T* out, typename realType<T>::Type* weights, unsigned int* indices,                        \
                     unsigned int* lower_bounds, unsigned int* upper_bounds, unsigned int num_elements,                \
                     unsigned int num_batches)                                                                         \
 {                                                                                                                     \
  typedef typename realType<T>::Type REAL;                                                                             \
  const unsigned int idx = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(1) *                          \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_range(2) *                \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) +                \
                           sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *                          \
                               sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) +                \
                           sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);                        \
                                                                                                                       \
  if (idx < num_elements * num_batches) {                                                                              \
                                                                                                                       \
   const unsigned int batch_no = idx / num_elements;                                                                   \
   const unsigned int idx_in_batch = idx - batch_no * num_elements;                                                    \
                                                                                                                       \
   const unsigned int lower_bound = lower_bounds[idx_in_batch];                                                        \
   const unsigned int upper_bound = upper_bounds[idx_in_batch];                                                        \
                                                                                                                       \
   T val = T(0);                                                                                                       \
                                                                                                                       \
   if (lower_bound > upper_bound || lower_bound >= (_get_num_neighbors<D>() * num_elements) ||                         \
       upper_bound >= (_get_num_neighbors<D>() * num_elements)) {                                                      \
                                                                                                                       \
    out[idx] = T(0);                                                                                                   \
    return;                                                                                                            \
   }                                                                                                                   \
                                                                                                                       \
   for (unsigned int i = lower_bound; i < upper_bound; i++) {                                                          \
    unsigned int in_idx = indices[i];                                                                                  \
    if (in_idx >= num_elements) {                                                                                      \
     val = T(0);                                                                                                       \
     continue;                                                                                                         \
    }                                                                                                                  \
    REAL weight = weights[i];                                                                                          \
    val += (in[in_idx + batch_no * num_elements] * weight);                                                            \
   }                                                                                                                   \
   out[idx] = val;                                                                                                     \
  }                                                                                                                    \
 }                                                                                                                     \
                                                                                                                       \
 template <class T, unsigned int D>                                                                                    \
 void cu##COMPONENT<T, D>::mult_MH(cuNDArray<T>* in, cuNDArray<T>* out, bool accumulate)                               \
 {                                                                                                                     \
  if (!in || !out) {                                                                                                   \
   throw cuda_error("cuResampleOperator::mult_MH(): illegal input/output array.");                                     \
  }                                                                                                                    \
                                                                                                                       \
  if (!this->preprocessed_) {                                                                                          \
   throw cuda_error("cuResampleOperator::mult_MH(): preprocessing has not been performed.");                           \
  }                                                                                                                    \
                                                                                                                       \
  cuNDArray<T> tmp;                                                                                                    \
  if (accumulate) {                                                                                                    \
   tmp = *out;                                                                                                         \
  }                                                                                                                    \
                                                                                                                       \
  unsigned int num_disp_vectors = this->get_number_of_displacement_vectors();                                          \
  int surplus = this->offsets_->get_number_of_dimensions() - D;                                                        \
                                                                                                                       \
  if (surplus == 1) {                                                                                                  \
   if (in->get_number_of_elements() != out->get_number_of_elements()) {                                                \
    throw cuda_error("cuResampleOperator::mult_MH(): in/out array dimensions mismatch (1).");                          \
   }                                                                                                                   \
   if ((in->get_number_of_elements() % num_disp_vectors) != 0) {                                                       \
    throw cuda_error("cuResampleOperator::mult_MH(): in/out array dimensions mismatch displacement field (1).");       \
   }                                                                                                                   \
  }                                                                                                                    \
                                                                                                                       \
  if (surplus == 2) {                                                                                                  \
   if ((in->get_number_of_elements() % out->get_number_of_elements()) != 0) {                                          \
    throw cuda_error("cuResampleOperator::mult_MH(): in/out array dimensions mismatch (2).");                          \
   }                                                                                                                   \
   if (in->get_number_of_dimensions() != (D + 1) || in->get_number_of_elements() != num_disp_vectors) {                \
    throw cuda_error("cuResampleOperator::mult_MH(): output array dimensions mismatch displacement field.");           \
   }                                                                                                                   \
  }                                                                                                                    \
                                                                                                                       \
  cuNDArray<T>* tmp_out = out; bool mod_out = false;                                                                   \
  if (surplus == 2 && (in->get_number_of_elements() / out->get_number_of_elements()) > 1) {                            \
   mod_out = true;                                                                                                     \
   tmp_out = new cuNDArray<T>(in->get_dimensions());                                                                   \
  }                                                                                                                    \
                                                                                                                       \
  typename uint64d<D>::Type matrix_size = from_std_vector<size_t, D>(this->offsets_->get_dimensions());                \
  unsigned int num_batches = (surplus == 2) ? 1 : in->get_number_of_elements() / prod(matrix_size);                    \
  unsigned int extended_dim = (surplus == 1) ? 1 : in->get_size(D);                                                    \
  unsigned int num_elements = prod(matrix_size) * extended_dim;                                                        \
                                                                                                                       \
  dpct::dim3 blockDim, gridDim;                                                                                        \
                                                                                                                       \
  setup_grid(num_elements, &blockDim, &gridDim, num_batches);                                                          \
  {                                                                                                                    \
   auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};       \
   dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});                        \
                                                                                                                       \
   dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {                                                         \
    auto in_get_data_ptr_ct0 = in->get_data_ptr();                                                                     \
    auto tmp_out_get_data_ptr_ct1 = tmp_out->get_data_ptr();                                                           \
    auto raw_pointer_cast_this_weights__ct2 = dpct::get_raw_pointer(&this->weights_[0]);                                    \
    auto raw_pointer_cast_this_indices__ct3 = dpct::get_raw_pointer(&this->indices_[0]);                                    \
    auto raw_pointer_cast_this_lower_bounds__ct4 = dpct::get_raw_pointer(&this->lower_bounds_[0]);                          \
    auto raw_pointer_cast_this_upper_bounds__ct5 = dpct::get_raw_pointer(&this->upper_bounds_[0]);                          \
    auto num_elements_ct6 = num_elements;                                                                              \
    auto num_batches_ct7 = num_batches;                                                                                \
                                                                                                                       \
    cgh.depends_on(dpct::get_current_device().get_in_order_queues_last_events());                                      \
                                                                                                                       \
    cgh.parallel_for<dpct_kernel_name<class mult_MH_kernel_9d1801, T, dpct_kernel_scalar<D>>>(                         \
        sycl::nd_range<3>(gridDim * blockDim, blockDim), exp_props, [=](sycl::nd_item<3> item_ct1) {                   \
         mult_MH_kernel<T, D>(in_get_data_ptr_ct0, tmp_out_get_data_ptr_ct1, raw_pointer_cast_this_weights__ct2,       \
                              raw_pointer_cast_this_indices__ct3, raw_pointer_cast_this_lower_bounds__ct4,             \
                              raw_pointer_cast_this_upper_bounds__ct5, num_elements_ct6, num_batches_ct7);             \
        });                                                                                                            \
   });                                                                                                                 \
  }                                                                                                                    \
                                                                                                                       \
  if (mod_out) {                                                                                                       \
   *out = *sum<T>(tmp_out, D);                                                                                         \
   delete tmp_out;                                                                                                     \
  }                                                                                                                    \
                                                                                                                       \
  CHECK_FOR_CUDA_ERROR();                                                                                              \
                                                                                                                       \
  if (accumulate) {                                                                                                    \
   *out += tmp;                                                                                                        \
  }                                                                                                                    \
 }
