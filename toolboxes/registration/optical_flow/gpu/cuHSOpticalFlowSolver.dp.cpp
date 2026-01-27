#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuHSOpticalFlowSolver.h"
#include "vector_td_utilities.h"
#include <cmath>

namespace Gadgetron{

  //
  // Kernel prototype declarations
  //

/* DPCT_ORIG   template<class REAL, unsigned int D> __global__
  void HornSchunk_kernel(const REAL*,const REAL*,const REAL*,REAL*,typename uintd<D>::Type,unsigned
  int,REAL,REAL,unsigned int*);*/
  template <class REAL, unsigned int D>
  void HornSchunk_kernel(const REAL*, const REAL*, const REAL*, REAL*, typename uintd<D>::Type, unsigned int, REAL,
                         REAL, unsigned int*, char* _shared_mem);

  //
  // Reference to shared memory
  //

/* DPCT_ORIG   extern __shared__ char _shared_mem[];*/

  //
  // Implementation
  //

  template <class T, unsigned int D>
  boost::shared_ptr<cuNDArray<T>> cuHSOpticalFlowSolver<T, D>::core_solver(cuNDArray<T>* gradient_image,
                                                                           cuNDArray<T>* stencil_image) try {
    // Sanity checks
    //
  
    if( !gradient_image ){
      throw std::runtime_error("cuHSOpticalFlowSolver::core_solver(): illegal input gradient image received.");
    }
  
    if( gradient_image->get_number_of_dimensions() <= D ){
      throw std::runtime_error("cuHSOpticalFlowSolver::core_solver(): number of gradient image dimensions is too small.");
    }
    
    // The dimensions of the displacement field should match the gradient field
    // - when removing the temporal gradient component (replacing D+1 with D)
    //
  
    std::vector<size_t> disp_dims = gradient_image->get_dimensions();
    disp_dims.pop_back(); disp_dims.push_back(D);

    boost::shared_ptr< cuNDArray<T> > displacements_ping(new cuNDArray<T>(disp_dims));
    boost::shared_ptr< cuNDArray<T> > displacements_pong(new cuNDArray<T>(disp_dims));
  
    clear(displacements_ping.get());
    clear(displacements_pong.get());
      
    // Setup grid
    //

    typename uint64d<D>::Type matrix_size = from_std_vector<size_t,D>( gradient_image->get_dimensions() );  
    unsigned int number_of_elements = prod(matrix_size);
    unsigned int number_of_batches = 1;
  
    for( unsigned int d=D; d<gradient_image->get_number_of_dimensions()-1; d++ ){
      number_of_batches *= gradient_image->get_size(d);
    }

/* DPCT_ORIG     dim3 blockDim; dim3 gridDim;*/
    dpct::dim3 blockDim; dpct::dim3 gridDim;
    this->setup_grid( &blockDim, &gridDim, number_of_elements, number_of_batches*D, true, D );
  
    // Allocate continuation flag (used for early Jacobi termination by the kernel)
    //
  
    unsigned int *continue_flag;
/* DPCT_ORIG     if( cudaMalloc((void**)&continue_flag, sizeof(unsigned int) ) != cudaSuccess ) {*/
    if (DPCT_CHECK_ERROR(continue_flag = sycl::malloc_device<unsigned int>(1, dpct::get_in_order_queue())) != 0) {
      throw std::runtime_error("cuHSOpticalFlowSolver::core_solver(): failed to allocate continuation flag.");
    }
  
    unsigned int iteration_no = 0;
    cuNDArray<T> *ping = displacements_ping.get();
    cuNDArray<T> *pong = displacements_pong.get();

    if( this->output_mode_ >= cuOpticalFlowSolver<T,D>::OUTPUT_VERBOSE ) {
      GDEBUG_STREAM(std::endl);
    }

    //
    // Main Jacobi loop
    //

    while(true){
    
      if( this->output_mode_ >= cuOpticalFlowSolver<T,D>::OUTPUT_VERBOSE ) {
        GDEBUG_STREAM("."; std::cerr.flush());
      }
    
      // Clear termination flag
      //
    
      unsigned int _continue_flag = 0;
/* DPCT_ORIG       if( cudaMemcpy( continue_flag, &_continue_flag, sizeof(unsigned int), cudaMemcpyHostToDevice ) !=
 * cudaSuccess ) {*/
      if (DPCT_CHECK_ERROR(
              dpct::get_in_order_queue().memcpy(continue_flag, &_continue_flag, sizeof(unsigned int)).wait()) != 0) {
        throw std::runtime_error("cuHSOpticalFlowSolver::core_solver(): failed to set continuation flag.");
      }
    
      // Invoke kernel
      //

/* DPCT_ORIG       HornSchunk_kernel<T,D><<< gridDim, blockDim, (blockDim.x*blockDim.y)*sizeof(T) >>>
        ( gradient_image->get_data_ptr(), (stencil_image) ? stencil_image->get_data_ptr() : 0x0,
          ping->get_data_ptr(), pong->get_data_ptr(),
          vector_td<unsigned int,D>(matrix_size), number_of_batches, alpha_, this->limit_*this->limit_, continue_flag
   );*/
      /*
      DPCT1049:105: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query
      info::device::max_work_group_size. Adjust the work-group size if needed.
      */
        /*
        DPCT1129:104: The type "vector_td<unsigned int, D>" is used in the SYCL kernel, but it is not device copyable.
        The sycl::is_device_copyable specialization has been added for this type. Please review the code.
        */
        {
            dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler& cgh) {
                sycl::local_accessor<char, 1> _shared_mem_acc_ct1(sycl::range<1>((blockDim.x * blockDim.y) * sizeof(T)),
                                                                  cgh);

                auto gradient_image_get_data_ptr_ct0 = gradient_image->get_data_ptr();
                auto stencil_image_stencil_image_get_data_ptr_ct1 =
                    (stencil_image) ? stencil_image->get_data_ptr() : 0x0;
                auto ping_get_data_ptr_ct2 = ping->get_data_ptr();
                auto pong_get_data_ptr_ct3 = pong->get_data_ptr();
                auto vector_td_unsigned_int_D_matrix_size_ct4 = vector_td<unsigned int, D>(matrix_size);
                auto alpha__ct6 = alpha_;
                auto this_limit__this_limit__ct7 = this->limit_ * this->limit_;

                cgh.parallel_for<dpct_kernel_name<class HornSchunk_kernel_84c9f1, T, dpct_kernel_scalar<D>>>(
                    sycl::nd_range<3>(gridDim * blockDim, blockDim), [=](sycl::nd_item<3> item_ct1) {
                        HornSchunk_kernel<T, D>(
                            gradient_image_get_data_ptr_ct0, stencil_image_stencil_image_get_data_ptr_ct1,
                            ping_get_data_ptr_ct2, pong_get_data_ptr_ct3, vector_td_unsigned_int_D_matrix_size_ct4,
                            number_of_batches, alpha__ct6, this_limit__this_limit__ct7, continue_flag,
                            _shared_mem_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
        }

      CHECK_FOR_CUDA_ERROR();

      // Swap in/out buffers
      //
    
      cuNDArray<T> *tmp = ping;
      ping = pong;
      pong = tmp;

      // Check termination criteria
      //

/* DPCT_ORIG       if( cudaMemcpy(&_continue_flag, continue_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost) !=
 * cudaSuccess ) {*/
      if (DPCT_CHECK_ERROR(
              dpct::get_in_order_queue().memcpy(&_continue_flag, continue_flag, sizeof(unsigned int)).wait()) != 0) {
        throw std::runtime_error("cuHSOpticalFlowSolver::core_solver(): failed to evaluate the continuation flag.");
      }
    
      if( _continue_flag == 0 ){
        if( this->output_mode_ >= cuOpticalFlowSolver<T,D>::OUTPUT_VERBOSE ) {
          GDEBUG_STREAM(std::endl << "Break after " << iteration_no+1 << " iterations" << std::endl);
        }
        break;
      }
    
      if( iteration_no > this->max_num_iterations_per_level_ ) 
        break;    
    
      iteration_no++;
    }

/* DPCT_ORIG     if( cudaFree(continue_flag) != cudaSuccess ) {*/
    if (DPCT_CHECK_ERROR(dpct::dpct_free(continue_flag, dpct::get_in_order_queue())) != 0) {
      throw std::runtime_error("cuHSOpticalFlowSolver::core_solver(): failed to free continuation flag.");
    }
    
    if( ping == displacements_ping.get() )   
      return displacements_ping;
    else
      return displacements_pong;
  }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  // Helpers
  //

/* DPCT_ORIG   template<unsigned int D> __device__
  bool is_border_pixel_for_stride( typename intd<D>::Type stride, typename uintd<D>::Type co, typename uintd<D>::Type
  dims )*/
  template <unsigned int D>
  SYCL_EXTERNAL bool is_border_pixel_for_stride(typename intd<D>::Type stride, typename uintd<D>::Type co,
                                                typename uintd<D>::Type dims)
  {
    for( unsigned int d=0; d<D; d++ ){
      if( stride.vec[d] == -1 ){
        if( co.vec[d] == 0 ){
          return true;
        }
      }
      else if( stride.vec[d] == 1 ){
        if( co.vec[d] == (dims.vec[d]-1) ){
          return true;
        }
      }
    }
    return false;
  }
  
  template<unsigned int i, unsigned int j> struct Pow
  {
    enum { Value = i*Pow<i,j-1>::Value };
  };
  
  template <unsigned int i> struct Pow<i,1>
  {
    enum { Value = i };
  };
  
  // Horn-Schunk / Jacobi iteration
  //

/* DPCT_ORIG   template<class REAL, unsigned int D> __global__ void
  HornSchunk_kernel( const REAL * __restrict__ gradient_image, const REAL * __restrict__ stencil_image,
                     const REAL * __restrict__ in_disp, REAL * __restrict__ out_disp,
                     typename uintd<D>::Type matrix_size, unsigned int num_batches,
                     REAL alpha, REAL disp_thresh_sqr, unsigned int * __restrict__ continue_signal )*/
  /*
DPCT1110:103: The total declared local variable size in device function HornSchunk_kernel exceeds 128 bytes and may
cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
  template <class REAL, unsigned int D>
  void HornSchunk_kernel(const REAL* __restrict__ gradient_image, const REAL* __restrict__ stencil_image,
                         const REAL* __restrict__ in_disp, REAL* __restrict__ out_disp,
                         typename uintd<D>::Type matrix_size, unsigned int num_batches, REAL alpha,
                         REAL disp_thresh_sqr, unsigned int* __restrict__ continue_signal, char* _shared_mem)
  {  
    
    // The overall flow dimension corresponding to this thread
/* DPCT_ORIG     const unsigned int dim = threadIdx.y;*/
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const unsigned int dim = item_ct1.get_local_id(1);

    // The thread idx relative to the flow dimension
/* DPCT_ORIG     const unsigned int idx = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x;*/
    const unsigned int idx = item_ct1.get_group(1) * item_ct1.get_group_range(2) * item_ct1.get_local_range(2) +
                             item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    // Number of elements per batch
    const unsigned int num_elements_per_batch = prod(matrix_size);
    
    // Number of elements per dim
    const unsigned int num_elements_per_dim = num_elements_per_batch*num_batches;
    
    // We use shared memory to hold the averaged displacements
    REAL *shared_mem = (REAL*) _shared_mem;
    
    //
    // Find the average velocities (shared memory)
    //
    
    // Batch idx (second slowest varying dimension)   
    const unsigned int batch_idx = idx/num_elements_per_batch;
    
    // Local index to the image (or batch in our terminology)
    const unsigned int idx_in_batch = idx-batch_idx*num_elements_per_batch;
    
    // All threads (even out-of-range ones) must reach the synchronization point below
    //
    
    bool legal_idx = (idx < num_elements_per_dim);
    
    if( legal_idx && stencil_image && stencil_image[idx_in_batch] > REAL(0) )
      legal_idx = false;
    
    if( legal_idx ){
      
      // Local co to the image
      const typename uintd<D>::Type co = idx_to_co( idx_in_batch, matrix_size );
      
      const typename intd<D>::Type zeros(0);
      const typename intd<D>::Type ones(1);
      const typename intd<D>::Type threes(3);
      
      const int num_neighbors = Pow<3,D>::Value;
      REAL num_contribs = REAL(0);
      
      // Idx local to the shared memory
/* DPCT_ORIG       const unsigned int shared_idx = threadIdx.y*blockDim.x+threadIdx.x;*/
      const unsigned int shared_idx = item_ct1.get_local_id(1) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

      shared_mem[shared_idx] = REAL(0);
      
      for( int i=0; i<num_neighbors; i++ ){
	
        // Find the stride of the neighbor {-1, 0, 1}^D
        const typename intd<D>::Type stride = idx_to_co( i, threes ) - ones;
	
        // Verify that the neighbor is not out of bounds (and not the thread itself)
        if( !is_border_pixel_for_stride<D>( stride, co, matrix_size ) && !(stride==zeros) ){
	  
          // Compute average of neighbors
          //
	  
          const unsigned int base_offset = dim*num_elements_per_dim + batch_idx*num_elements_per_batch;
          const unsigned int neighbor_idx = (unsigned int) co_to_idx( vector_td<int,D>(co)+stride, vector_td<int,D>(matrix_size)) + base_offset;
	  
          shared_mem[shared_idx] += in_disp[neighbor_idx];
          num_contribs += REAL(1);
        }
      }
      
      // Normalize
      shared_mem[shared_idx] /= num_contribs;       	
    }
    
    // Block until all averages have been computed (we need all d dims below)
/* DPCT_ORIG     __syncthreads();*/
    /*
    DPCT1065:253: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to
    global memory.
    */
    item_ct1.barrier();

    if( legal_idx ){
      
      //
      // Update displacement field (Jacobi iteration)
      //
      
      REAL phi = REAL(0);
      REAL norm = REAL(0);
      
      typename reald<REAL,D>::Type derivatives;
      
      // Contributions from the spatial dimensions
      //
      
      for( unsigned int d=0; d<D; d++ ){
        derivatives.vec[d] = gradient_image[d*num_elements_per_dim+idx];
/* DPCT_ORIG         const unsigned int shared_idx = d*blockDim.x+threadIdx.x;*/
        const unsigned int shared_idx = d * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
        phi += (shared_mem[shared_idx]*derivatives.vec[d]);
        norm += (derivatives.vec[d]*derivatives.vec[d]);
      }
      
      // Contributions from the temporal dimension
      //
      
      phi += gradient_image[D*num_elements_per_dim+idx];
      
      // Normalize
      //
      
      phi /= (alpha*alpha+norm);
      
      // Form result displacement
      //

/* DPCT_ORIG       const unsigned int shared_idx = dim*blockDim.x+threadIdx.x;*/
      const unsigned int shared_idx = dim * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      REAL result = shared_mem[shared_idx]-derivatives.vec[dim]*phi;
      
      // Clear the "termination" flag if the displacement field has changed above the threshold
      //
      
      REAL delta = result-in_disp[dim*num_elements_per_dim+idx];
      if( delta*delta > disp_thresh_sqr )
        continue_signal[0] = 1;
      
      // Output result
      //
      
      out_disp[dim*num_elements_per_dim+idx] = result;
    }
  }
  
  // 
  // Template instantiation
  //
  
  template class EXPORTGPUREG cuHSOpticalFlowSolver<float,1>;
  template class EXPORTGPUREG cuHSOpticalFlowSolver<float,2>;
  template class EXPORTGPUREG cuHSOpticalFlowSolver<float,3>;
  template class EXPORTGPUREG cuHSOpticalFlowSolver<float,4>;
  
  template class EXPORTGPUREG cuHSOpticalFlowSolver<double,1>;
  template class EXPORTGPUREG cuHSOpticalFlowSolver<double,2>;
  template class EXPORTGPUREG cuHSOpticalFlowSolver<double,3>;
  template class EXPORTGPUREG cuHSOpticalFlowSolver<double,4>;
}
