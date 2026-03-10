/** \file cuNDArray.h
\brief GPU-based N-dimensional array (data container)
*/

#pragma once
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "core_defines.h"
#include "NDArray.h"
#include "hoNDArray.h"
#include "complext.h"
#include "GadgetronCuException.h"
#include "check_CUDA.h"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <dpct/dpl_utils.hpp>

namespace Gadgetron{

    template <typename T> class cuNDArray : public NDArray<T>
    {

    public:

        // Constructors
        //

        cuNDArray();
        cuNDArray(const cuNDArray<T> &a);
        explicit cuNDArray(const hoNDArray<T> &a);

#if __cplusplus > 199711L
        // Move constructor
        cuNDArray(cuNDArray<T>&& a);
#endif
        explicit cuNDArray(const std::vector<size_t> &dimensions);
        cuNDArray(const std::vector<size_t> &dimensions, int device_no);
        cuNDArray(const std::vector<size_t> &dimensions, T* data, bool delete_data_on_destruct = false);

        explicit cuNDArray(size_t len);
        cuNDArray(size_t sx, size_t sy);
        cuNDArray(size_t sx, size_t sy, size_t sz);
        cuNDArray(size_t sx, size_t sy, size_t sz, size_t st);
        cuNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp);
        cuNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq);
        cuNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr);
        cuNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss);

        // Destructor
        virtual ~cuNDArray();

        // Assignment operator
        cuNDArray<T>& operator=(const cuNDArray<T>& rhs);

#if __cplusplus > 199711L
        cuNDArray<T>& operator=(cuNDArray<T>&& rhs);
#endif
        cuNDArray<T>& operator=(const hoNDArray<T>& rhs);

        virtual void create(const std::vector<size_t> &dimensions);
        virtual void create(const std::vector<size_t> &dimensions, int device_no);
        virtual void create(const std::vector<size_t> &dimensions, T* data, bool delete_data_on_destruct = false);

        virtual void create(size_t len);
        virtual void create(size_t sx, size_t sy);
        virtual void create(size_t sx, size_t sy, size_t sz);
        virtual void create(size_t sx, size_t sy, size_t sz, size_t st);
        virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp);
        virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq);
        virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr);
        virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss);

        virtual boost::shared_ptr< hoNDArray<T> > to_host() const;
        virtual void to_host( hoNDArray<T> *out ) const;

        virtual void set_device(int device);
        int get_device();

        dpct::device_pointer<T> get_device_ptr();
        const dpct::device_pointer<T> get_device_ptr() const;
        dpct::device_pointer<T> begin();
        dpct::device_pointer<T> end();
        const dpct::device_pointer<T> begin() const;
        const dpct::device_pointer<T> end() const;

        T at( size_t idx );
        T operator[]( size_t idx );


    protected:

        int device_; 

        virtual void allocate_memory();
        virtual void deallocate_memory();
    };

    template <typename T> 
    cuNDArray<T>::cuNDArray() : Gadgetron::NDArray<T>::NDArray() 
    {
        this->device_ = dpct::get_current_device_id();
    }

    template <typename T>
    cuNDArray<T>::cuNDArray(const cuNDArray<T> &a) try
        : Gadgetron::NDArray<T>::NDArray()
    {
        this->device_ = dpct::get_current_device_id();
        this->data_ = 0;
        this->dimensions_ = a.dimensions_;
        allocate_memory();
        if (a.device_ == this->device_) {
            /*
            DPCT1064:31: Migrated cudaMemcpy call is used in a macro/template
            definition and may not be valid for all macro/template uses. Adjust
            the code.
            */
            CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                this->data_, a.data_, this->elements_ * sizeof(T))));
        } else {
            //This memory is on a different device, we must move it.
            /*
            DPCT1093:15: The "a.device_" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            dpct::select_device(a.device_);
            boost::shared_ptr< hoNDArray<T> > tmp = a.to_host();
            /*
            DPCT1093:16: The "this->device_" device may be not the one intended
            for use. Adjust the selected device if needed.
            */
            dpct::select_device(this->device_);
            dpct::err0 err =
                DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                     .memcpy(this->data_, tmp->get_data_ptr(),
                                             this->elements_ * sizeof(T))
                                     .wait());
            /*
            DPCT1000:18: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (err != 0) {
                /*
                DPCT1001:17: The statement could not be removed.
                */
                deallocate_memory();
                this->data_ = 0;
                this->dimensions_.clear();
                throw cuda_error(err);
            }
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

#if __cplusplus > 199711L
    template <typename T>
    cuNDArray<T>::cuNDArray(cuNDArray<T>&& a) : Gadgetron::NDArray<T>::NDArray()
    {
        device_ = a.device_;
        this->data_ = a.data_;
        this->dimensions_ = a.dimensions_;
        this->elements_ = a.elements_;
        a.data_=nullptr;
        this->delete_data_on_destruct_ = a.delete_data_on_destruct_;
    }
#endif
    template <typename T>
    cuNDArray<T>::cuNDArray(const hoNDArray<T> &a) try
        : Gadgetron::NDArray<T>::NDArray()
    {
        this->device_ = dpct::get_current_device_id();
        a.get_dimensions(this->dimensions_);
        allocate_memory();
        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(this->data_, a.get_data_ptr(),
                                         this->elements_ * sizeof(T))
                                 .wait()) != 0) {
            deallocate_memory();
            this->data_ = 0;
            this->dimensions_.clear();
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(const std::vector<size_t> &dimensions) : Gadgetron::NDArray<T>::NDArray()
    {
        this->device_ = dpct::get_current_device_id();
        create(dimensions);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(const std::vector<size_t> &dimensions, int device_no) : Gadgetron::NDArray<T>::NDArray()
    {
        this->device_ = dpct::get_current_device_id();
        create(dimensions,device_no);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(const std::vector<size_t> &dimensions, T* data, bool delete_data_on_destruct) : Gadgetron::NDArray<T>::NDArray()
    {
        this->device_ = dpct::get_current_device_id();
        create(dimensions,data,delete_data_on_destruct);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(size_t len)
    {
        std::vector<size_t> dim(1);
        dim[0] = len;
        this->device_ = dpct::get_current_device_id();
        create(dim);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(size_t sx, size_t sy)
    {
        std::vector<size_t> dim(2);
        dim[0] = sx;
        dim[1] = sy;
        this->device_ = dpct::get_current_device_id();
        create(dim);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(size_t sx, size_t sy, size_t sz)
    {
        std::vector<size_t> dim(3);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        this->device_ = dpct::get_current_device_id();
        create(dim);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(size_t sx, size_t sy, size_t sz, size_t st)
    {
        std::vector<size_t> dim(4);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        this->device_ = dpct::get_current_device_id();
        create(dim);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp)
    {
        std::vector<size_t> dim(5);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        this->device_ = dpct::get_current_device_id();
        create(dim);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq)
    {
        std::vector<size_t> dim(6);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        this->device_ = dpct::get_current_device_id();
        create(dim);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr)
    {
        std::vector<size_t> dim(7);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        this->device_ = dpct::get_current_device_id();
        create(dim);
    }

    template <typename T> 
    cuNDArray<T>::cuNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss)
    {
        std::vector<size_t> dim(8);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        dim[7] = ss;
        this->device_ = dpct::get_current_device_id();
        create(dim);
    }

    template <typename T> 
    cuNDArray<T>:: ~cuNDArray()
    { 
        if (this->delete_data_on_destruct_) 
            deallocate_memory();  
    }

#if __cplusplus > 199711L
    template <typename T>
    cuNDArray<T>& cuNDArray<T>::operator=(cuNDArray<T>&& rhs){

        if (&rhs == this) return *this;
        this->clear();
        this->dimensions_ = rhs.dimensions_;
        this->elements_ = rhs.elements_;
        device_ = rhs.device_;
        this->data_ = rhs.data_;
        rhs.data_ = nullptr;
        this->delete_data_on_destruct_ = rhs.delete_data_on_destruct_;
        return *this;
    }
#endif

    template <typename T>
    cuNDArray<T> &cuNDArray<T>::operator=(const cuNDArray<T> &rhs) try {
        int cur_device;
        CUDA_CALL(DPCT_CHECK_ERROR(cur_device = dpct::get_current_device_id()));
        bool dimensions_match = this->dimensions_equal(rhs);
        if (dimensions_match && (rhs.device_ == cur_device) && (cur_device == this->device_)) {
            /*
            DPCT1064:32: Migrated cudaMemcpy call is used in a macro/template
            definition and may not be valid for all macro/template uses. Adjust
            the code.
            */
            CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                this->data_, rhs.data_, this->elements_ * sizeof(T))));
        }
        else {
            /*
            DPCT1093:19: The "this->device_" device may be not the one intended
            for use. Adjust the selected device if needed.
            */
            CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(this->device_)));
            if( !dimensions_match ){
                deallocate_memory();
                this->elements_ = rhs.elements_;
                this->dimensions_ = rhs.dimensions_;
                allocate_memory();
            }
            if (this->device_ == rhs.device_) {
                if (DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                        this->data_, rhs.data_, this->elements_ * sizeof(T))) !=
                    0) {
                    /*
                    DPCT1093:20: The "cur_device" device may be not the one
                    intended for use. Adjust the selected device if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: failed to copy data (2)");
                }
            } else {
                /*
                DPCT1093:21: The "rhs.device_" device may be not the one
                intended for use. Adjust the selected device if needed.
                */
                if (DPCT_CHECK_ERROR(dpct::select_device(rhs.device_)) != 0) {
                    /*
                    DPCT1093:22: The "cur_device" device may be not the one
                    intended for use. Adjust the selected device if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: unable to set device no (2)");
                }
                boost::shared_ptr< hoNDArray<T> > tmp = rhs.to_host();
                /*
                DPCT1093:23: The "this->device_" device may be not the one
                intended for use. Adjust the selected device if needed.
                */
                if (DPCT_CHECK_ERROR(dpct::select_device(this->device_)) != 0) {
                    /*
                    DPCT1093:24: The "cur_device" device may be not the one
                    intended for use. Adjust the selected device if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: unable to set device no (3)");
                }
                if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(this->data_,
                                                 tmp->get_data_ptr(),
                                                 this->elements_ * sizeof(T))
                                         .wait()) != 0) {
                    /*
                    DPCT1093:25: The "cur_device" device may be not the one
                    intended for use. Adjust the selected device if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: failed to copy data (3)");
                }
            }
            /*
            DPCT1093:26: The "cur_device" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            if (DPCT_CHECK_ERROR(dpct::select_device(cur_device)) != 0) {
                throw cuda_error("cuNDArray::operator=: unable to restore to current device");
            }
        }
        return *this;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T>
    cuNDArray<T> &cuNDArray<T>::operator=(const hoNDArray<T> &rhs) try {
        int cur_device;
        CUDA_CALL(DPCT_CHECK_ERROR(cur_device = dpct::get_current_device_id()));
        bool dimensions_match = this->dimensions_equal(rhs);
        if (dimensions_match && (cur_device == this->device_)) {
            /*
            DPCT1064:33: Migrated cudaMemcpy call is used in a macro/template
            definition and may not be valid for all macro/template uses. Adjust
            the code.
            */
            CUDA_CALL(DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memcpy(this->get_data_ptr(), rhs.get_data_ptr(),
                            this->get_number_of_elements() * sizeof(T))
                    .wait()));
        }
        else {
            /*
            DPCT1093:27: The "this->device_" device may be not the one intended
            for use. Adjust the selected device if needed.
            */
            CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(this->device_)));
            if( !dimensions_match ){
                deallocate_memory();
                this->elements_ = rhs.get_number_of_elements();
                rhs.get_dimensions(this->dimensions_);
                allocate_memory();
            }
            if (DPCT_CHECK_ERROR(
                    dpct::get_in_order_queue()
                        .memcpy(this->get_data_ptr(), rhs.get_data_ptr(),
                                this->get_number_of_elements() * sizeof(T))
                        .wait()) != 0) {
                    /*
                    DPCT1093:28: The "cur_device" device may be not the one
                    intended for use. Adjust the selected device if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: failed to copy data (1)");
            }
            /*
            DPCT1093:29: The "cur_device" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            if (DPCT_CHECK_ERROR(dpct::select_device(cur_device)) != 0) {
                throw cuda_error("cuNDArray::operator=: unable to restore to current device");
            }
        }
        return *this;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(const std::vector<size_t> &dimensions)
    {
        if ( this->dimensions_equal(dimensions) )
        {
            return;
        }

        return Gadgetron::NDArray<T>::create(dimensions);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(const std::vector<size_t> &dimensions, int device_no)
    {
        if (device_no < 0){
            throw cuda_error("cuNDArray::create: illegal device no");
        }

        if ( this->dimensions_equal(dimensions) && this->device_==device_no )
        {
            return;
        }

        this->device_ = device_no; 
        Gadgetron::NDArray<T>::create(dimensions);
    }

    template <typename T>
    inline void cuNDArray<T>::create(const std::vector<size_t> &dimensions,
                                     T *data,
                                     bool delete_data_on_destruct) try {
        if (!data) {
            throw std::runtime_error("cuNDArray::create: 0x0 pointer provided");
        }

        int tmp_device;
        if (DPCT_CHECK_ERROR(tmp_device = dpct::get_current_device_id()) != 0) {
            throw cuda_error("cuNDArray::create: Unable to query for device");
        }

        dpct::device_info deviceProp;
        if (DPCT_CHECK_ERROR(
                dpct::get_device(tmp_device).get_device_info(deviceProp)) !=
            0) {
            throw cuda_error("cuNDArray::create: Unable to query device properties");
        }

        if (deviceProp.get_host_unified_memory()) {
            dpct::pointer_attributes attrib;
            if (DPCT_CHECK_ERROR(attrib.init(data)) != 0) {
                CHECK_FOR_CUDA_ERROR();
                throw cuda_error("cuNDArray::create: Unable to determine attributes of pointer");
            }
            this->device_ = attrib.get_device_id();
        } else {
            this->device_ = tmp_device;
        }

        Gadgetron::NDArray<T>::create(dimensions, data, delete_data_on_destruct);
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(size_t len)
    {
        std::vector<size_t> dim(1);
        dim[0] = len;
        this->create(dim);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(size_t sx, size_t sy)
    {
        std::vector<size_t> dim(2);
        dim[0] = sx;
        dim[1] = sy;
        this->create(dim);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(size_t sx, size_t sy, size_t sz)
    {
        std::vector<size_t> dim(3);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        this->create(dim);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st)
    {
        std::vector<size_t> dim(4);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        this->create(dim);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp)
    {
        std::vector<size_t> dim(5);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        this->create(dim);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq)
    {
        std::vector<size_t> dim(6);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        this->create(dim);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr)
    {
        std::vector<size_t> dim(7);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        this->create(dim);
    }

    template <typename T> 
    inline void cuNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss)
    {
        std::vector<size_t> dim(8);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        dim[7] = ss;
        this->create(dim);
    }

    template <typename T> 
    inline boost::shared_ptr< hoNDArray<T> > cuNDArray<T>::to_host() const
    {
      boost::shared_ptr< hoNDArray<T> > ret(new hoNDArray<T>());
      this->to_host(ret.get());
      return ret;
    }

    template <typename T>
    inline void cuNDArray<T>::to_host(hoNDArray<T> *out) const try {
        if( !out ){
            throw std::runtime_error("cuNDArray::to_host(): illegal array passed.");
        }

        if( out->get_number_of_elements() != this->get_number_of_elements() ){	
            out->create(this->get_dimensions());
        }

        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(out->get_data_ptr(), this->data_,
                                         this->elements_ * sizeof(T))
                                 .wait()) != 0) {
            throw cuda_error("cuNDArray::to_host(): failed to copy memory from device");
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T> inline void cuNDArray<T>::set_device(int device) try {
        if( device_ == device )
            return;

        int cur_device;
        if (DPCT_CHECK_ERROR(cur_device = dpct::get_current_device_id()) != 0) {
            throw cuda_error("cuNDArray::set_device: unable to get device no");
        }

        /*
        DPCT1093:4: The "device_" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        if (cur_device != device_ &&
            DPCT_CHECK_ERROR(dpct::select_device(device_)) != 0) {
            throw cuda_error("cuNDArray::set_device: unable to set device no");
        }

        boost::shared_ptr< hoNDArray<T> > tmp = to_host();
        deallocate_memory();
        /*
        DPCT1093:5: The "device" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        if (DPCT_CHECK_ERROR(dpct::select_device(device)) != 0) {
            /*
            DPCT1093:6: The "cur_device" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            dpct::select_device(cur_device);
            throw cuda_error("cuNDArray::set_device: unable to set device no (2)");
        }

        device_ = device;
        allocate_memory();
        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(this->data_, tmp->get_data_ptr(),
                                         this->elements_ * sizeof(T))
                                 .wait()) != 0) {
            /*
            DPCT1093:7: The "cur_device" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            dpct::select_device(cur_device);
            throw cuda_error("cuNDArray::set_device: failed to copy data");
        }

        /*
        DPCT1093:8: The "cur_device" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        if (DPCT_CHECK_ERROR(dpct::select_device(cur_device)) != 0) {
            throw cuda_error("cuNDArray::set_device: unable to restore device to current device");
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T> 
    inline int cuNDArray<T>::get_device() { return device_; }

    template <typename T>
    inline dpct::device_pointer<T> cuNDArray<T>::get_device_ptr()
    {
        return dpct::device_pointer<T>(this->data_);
    }

    template <typename T>
    inline const dpct::device_pointer<T> cuNDArray<T>::get_device_ptr() const
    {
        return dpct::device_pointer<T>(this->data_);
    }

    template <typename T> inline dpct::device_pointer<T> cuNDArray<T>::begin()
    {
        return dpct::device_pointer<T>(this->data_);
    }

    template <typename T> inline dpct::device_pointer<T> cuNDArray<T>::end()
    {
        return dpct::device_pointer<T>(this->data_) +
               this->get_number_of_elements();
    }

    template <typename T>
    inline const dpct::device_pointer<T> cuNDArray<T>::begin() const
    {
        return dpct::device_pointer<T>(this->data_);
    }

    template <typename T>
    inline const dpct::device_pointer<T> cuNDArray<T>::end() const
    {
        return dpct::device_pointer<T>(this->data_) +
               this->get_number_of_elements();
    }

    template <typename T>
    inline T cuNDArray<T>::at( size_t idx )
    {
        if( idx >= this->get_number_of_elements() ){
            throw std::runtime_error("cuNDArray::at(): index out of range.");
        }
        T res;
        /*
        DPCT1064:34: Migrated cudaMemcpy call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        CUDA_CALL(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(&res, &this->get_data_ptr()[idx], sizeof(T))
                .wait()));
        return res;
    }

    template <typename T> 
    inline T cuNDArray<T>::operator[]( size_t idx )
    {
        if( idx >= this->get_number_of_elements() ){
            throw std::runtime_error("cuNDArray::operator[]: index out of range.");
        }
        T res;
        /*
        DPCT1064:35: Migrated cudaMemcpy call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        CUDA_CALL(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(&res, &this->get_data_ptr()[idx], sizeof(T))
                .wait()));
        return res;
    }

    template <typename T> void cuNDArray<T>::allocate_memory() try {
        deallocate_memory();

        this->elements_ = 1;
        if (this->dimensions_.empty())
            throw std::runtime_error("cuNDArray::allocate_memory() : dimensions is empty.");
        for (size_t i = 0; i < this->dimensions_.size(); i++) {
            this->elements_ *= this->dimensions_[i];
        } 

        size_t size = this->elements_ * sizeof(T);

        int device_no_old;
        if (DPCT_CHECK_ERROR(device_no_old = dpct::get_current_device_id()) != 0) {
            throw cuda_error("cuNDArray::allocate_memory: unable to get device no");
        }

        if (device_ != device_no_old) {
            /*
            DPCT1093:9: The "device_" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            if (DPCT_CHECK_ERROR(dpct::select_device(device_)) != 0) {
                throw cuda_error("cuNDArray::allocate_memory: unable to set device no");
            }
        }

        if (DPCT_CHECK_ERROR((this->data_) = (typename std::remove_reference<
                                              decltype(this->data_)>::type)
                                 sycl::malloc_device(
                                     size, dpct::get_in_order_queue())) != 0) {
            size_t free = 0, total = 0;
            /*
            DPCT1106:30: 'cudaMemGetInfo' was migrated with the Intel extensions
            for device information which may not be supported by all compilers
            or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(free, total);
            std::stringstream err("cuNDArray::allocate_memory() : Error allocating CUDA memory");
            err << "CUDA Memory: " << free << " (" << total << ")";

            err << "   memory requested: " << size << "( ";
            for (size_t i = 0; i < this->dimensions_.size(); i++) {
                std::cerr << this->dimensions_[i] << " ";
            }
            err << ")";
            this->data_ = 0;
            throw std::runtime_error(err.str());
        }

        if (device_ != device_no_old) {
            /*
            DPCT1093:10: The "device_no_old" device may be not the one intended
            for use. Adjust the selected device if needed.
            */
            if (DPCT_CHECK_ERROR(dpct::select_device(device_no_old)) != 0) {
                throw cuda_error("cuNDArray::allocate_memory: unable to restore device no");
            }
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T> void cuNDArray<T>::deallocate_memory() try {
        if (this->data_) {

            int device_no_old;
            CUDA_CALL(DPCT_CHECK_ERROR(device_no_old =
                                           dpct::get_current_device_id()));
            if (device_ != device_no_old) {
                /*
                DPCT1093:13: The "device_" device may be not the one intended
                for use. Adjust the selected device if needed.
                */
                CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(device_)));
            }

            /*
            DPCT1064:36: Migrated cudaFree call is used in a macro/template
            definition and may not be valid for all macro/template uses. Adjust
            the code.
            */
            CUDA_CALL(DPCT_CHECK_ERROR(
                dpct::dpct_free(this->data_, dpct::get_in_order_queue())));
            if (device_ != device_no_old) {
                /*
                DPCT1093:14: The "device_no_old" device may be not the one
                intended for use. Adjust the selected device if needed.
                */
                CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(device_no_old)));
            }
            this->data_ = 0;
        }
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
}
