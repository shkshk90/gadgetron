/** \file cuNDArray.h
\brief GPU-based N-dimensional array (data container)
*/

#pragma once

#include "NDArray.h"
#include "hoNDArray.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include <memory>
#include <vector>
#include <cstdint>
#include <sstream>


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

        virtual std::shared_ptr< hoNDArray<T> > to_host() const;
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
    cuNDArray<T>::cuNDArray(const cuNDArray<T> &a) : Gadgetron::NDArray<T>::NDArray() 
    {
        this->device_ = dpct::get_current_device_id();
        this->data_ = 0;
        this->dimensions_ = a.dimensions_;
        allocate_memory();
        if (a.device_ == this->device_) {
            dpct::get_in_order_queue().memcpy(this->data_, a.data_, this->elements_ * sizeof(T));
        } else {
            //This memory is on a different device, we must move it.
            dpct::select_device(a.device_);
            std::shared_ptr< hoNDArray<T> > tmp = a.to_host();
            dpct::select_device(this->device_);

            try {
                dpct::get_in_order_queue().memcpy(this->data_, tmp->get_data_ptr(), this->elements_ * sizeof(T)).wait();
            } catch ( ... ) {
                deallocate_memory();
                this->data_ = 0;
                this->dimensions_.clear();
                throw;
            }
        }
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
    cuNDArray<T>::cuNDArray(const hoNDArray<T> &a) : Gadgetron::NDArray<T>::NDArray() 
    {
        this->device_ = dpct::get_current_device_id();
        a.get_dimensions(this->dimensions_);
        allocate_memory();
        try {
            dpct::get_in_order_queue().memcpy(this->data_, a.get_data_ptr(), this->elements_ * sizeof(T)).wait()
        } catch ( ... ) {
            deallocate_memory();
            this->data_ = 0;
            this->dimensions_.clear();
        }
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

    template <typename T> cuNDArray<T>& cuNDArray<T>::operator=(const cuNDArray<T>& rhs) try {
        int cur_device;

        cur_device = dpct::get_current_device_id();
        bool dimensions_match = this->dimensions_equal(rhs);
        if (dimensions_match && (rhs.device_ == cur_device) && (cur_device == this->device_)) {
            dpct::get_in_order_queue().memcpy(this->data_, rhs.data_, this->elements_ * sizeof(T));
        }
        else {
            /*
            DPCT1093:4: The "this->device_" device may be not the one intended for use. Adjust the selected device if
            needed.
            */

            dpct::select_device(this->device_);
            if( !dimensions_match ){
                deallocate_memory();
                this->elements_ = rhs.elements_;
                this->dimensions_ = rhs.dimensions_;
                allocate_memory();
            }
            if (this->device_ == rhs.device_) {
                try {
                    dpct::get_in_order_queue().memcpy(this->data_, rhs.data_, this->elements_ * sizeof(T));
                } catch ( ... ) {
                    dpct::select_device(cur_device);
                    throw;
                }
            } else {
                /*
                DPCT1093:6: The "rhs.device_" device may be not the one intended for use. Adjust the selected device if
                needed.
                */
                try {
                    dpct::select_device(rhs.device_);
                }
                catch (const std::exception& e) {
                    dpct::select_device(cur_device);
                    throw;
                }
                std::shared_ptr< hoNDArray<T> > tmp = rhs.to_host();
                
                try {
                    dpct::select_device(this->device_);
                }
                catch (const std::exception& e) {
                    dpct::select_device(cur_device);
                    throw;
                }
                try {
                    dpct::get_in_order_queue()
                                         .memcpy(this->data_, tmp->get_data_ptr(), this->elements_ * sizeof(T))
                                         .wait()
                } catch ( ... ) {
                    dpct::select_device(cur_device);
                    throw;
                }
            }
            dpct::select_device(cur_device);
        }
        return *this;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      throw;
    }

    template <typename T> cuNDArray<T>& cuNDArray<T>::operator=(const hoNDArray<T>& rhs) try {
        int cur_device;

        cur_device = dpct::get_current_device_id();
        bool dimensions_match = this->dimensions_equal(rhs);
        if (dimensions_match && (cur_device == this->device_)) {
            dpct::get_in_order_queue()
                    .memcpy(this->get_data_ptr(), rhs.get_data_ptr(), this->get_number_of_elements() * sizeof(T))
                    .wait();
        }
        else {
            /*
            DPCT1093:12: The "this->device_" device may be not the one intended for use. Adjust the selected device if
            needed.
            */

            dpct::select_device(this->device_);
            if( !dimensions_match ){
                deallocate_memory();
                this->elements_ = rhs.get_number_of_elements();
                rhs.get_dimensions(this->dimensions_);
                allocate_memory();
            }
            try {
                dpct::get_in_order_queue()
                        .memcpy(this->get_data_ptr(), rhs.get_data_ptr(), this->get_number_of_elements() * sizeof(T))
                        .wait();
            } catch ( ... ) {
                    dpct::select_device(cur_device);
                    throw;
            }
            /*
            DPCT1093:14: The "cur_device" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            dpct::select_device(cur_device);
        }
        return *this;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      throw;
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
            throw std::runtime_error("cuNDArray::create: illegal device no");
        }

        if ( this->dimensions_equal(dimensions) && this->device_==device_no )
        {
            return;
        }

        this->device_ = device_no; 
        Gadgetron::NDArray<T>::create(dimensions);
    }

    template <typename T>
    inline void cuNDArray<T>::create(const std::vector<size_t>& dimensions, T* data, bool delete_data_on_destruct) try {
        if (!data) {
            throw std::runtime_error("cuNDArray::create: 0x0 pointer provided");
        }

        int tmp_device;
        tmp_device = dpct::get_current_device_id();

        dpct::device_info deviceProp;
        dpct::get_device(tmp_device).get_device_info(deviceProp);

        if (deviceProp.get_host_unified_memory()) {
            dpct::pointer_attributes attrib;
            if (DPCT_CHECK_ERROR(attrib.init(data)) != 0) {
                throw std::runtime_error("cuNDArray::create: Unable to determine attributes of pointer");
            }
            this->device_ = attrib.get_device_id();
        } else {
            this->device_ = tmp_device;
        }

        Gadgetron::NDArray<T>::create(dimensions, data, delete_data_on_destruct);
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      throw;
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
    inline std::shared_ptr< hoNDArray<T> > cuNDArray<T>::to_host() const
    {
      std::shared_ptr< hoNDArray<T> > ret(new hoNDArray<T>());
      this->to_host(ret.get());
      return ret;
    }

    template <typename T> 
    inline void cuNDArray<T>::to_host( hoNDArray<T> *out ) const 
    {
        if( !out ){
            throw std::runtime_error("cuNDArray::to_host(): illegal array passed.");
        }

        if( out->get_number_of_elements() != this->get_number_of_elements() ){	
            out->create(this->get_dimensions());
        }

        try { 
            dpct::get_in_order_queue()
                .memcpy(out->get_data_ptr(), this->data_, this->elements_ * sizeof(T))
                .wait();
        } catch ( ... ) {
            throw runtime_error("cuNDArray::to_host(): failed to copy memory from device");
        }
    }

    template <typename T> inline void cuNDArray<T>::set_device(int device) try {
        if( device_ == device )
            return;

        int cur_device;
        cur_device = dpct::get_current_device_id();

        /*
        DPCT1093:15: The "device_" device may be not the one intended for use. Adjust the selected device if needed.
        */
        if (cur_device != device_) {
            try {
                dpct::select_device(device_);
            } catch ( ... ) {
                throw runtime_error("cuNDArray::set_device: unable to set device no");
            }
        }

        std::shared_ptr< hoNDArray<T> > tmp = to_host();
        deallocate_memory();
        
        try {
            dpct::select_device(device);
        } catch (const std::exception& ) {
            dpct::select_device(cur_device);
            throw;
        }

        device_ = device;
        allocate_memory();
        try {
            dpct::get_in_order_queue()
                .memcpy(this->data_, tmp->get_data_ptr(), this->elements_ * sizeof(T))
                .wait();
        } catch ( ... ) {
            dpct::select_device(cur_device);
            throw runtime_error("cuNDArray::set_device: failed to copy data");
        }

        /*
        DPCT1093:19: The "cur_device" device may be not the one intended for use. Adjust the selected device if needed.
        */
        dpct::select_device(cur_device);
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      throw;
    }

    template <typename T> 
    inline int cuNDArray<T>::get_device() { return device_; }

    template <typename T> inline dpct::device_pointer<T> cuNDArray<T>::get_device_ptr()
    {
        return dpct::device_pointer<T>(this->data_);
    }

    template <typename T> inline const dpct::device_pointer<T> cuNDArray<T>::get_device_ptr() const
    {
        return dpct::device_pointer<T>(this->data_);
    }

    template <typename T> inline dpct::device_pointer<T> cuNDArray<T>::begin()
    {
        return dpct::device_pointer<T>(this->data_);
    }

    template <typename T> inline dpct::device_pointer<T> cuNDArray<T>::end()
    {
        return dpct::device_pointer<T>(this->data_) + this->get_number_of_elements();
    }

    template <typename T> inline const dpct::device_pointer<T> cuNDArray<T>::begin() const
    {
        return dpct::device_pointer<T>(this->data_);
    }

    template <typename T> inline const dpct::device_pointer<T> cuNDArray<T>::end() const
    {
        return dpct::device_pointer<T>(this->data_) + this->get_number_of_elements();
    }

    template <typename T>
    inline T cuNDArray<T>::at( size_t idx )
    {
        if( idx >= this->get_number_of_elements() ){
            throw std::runtime_error("cuNDArray::at(): index out of range.");
        }
        T res;

        dpct::get_in_order_queue().memcpy(&res, &this->get_data_ptr()[idx], sizeof(T)).wait();
        return res;
    }

    template <typename T> 
    inline T cuNDArray<T>::operator[]( size_t idx )
    {
        if( idx >= this->get_number_of_elements() ){
            throw std::runtime_error("cuNDArray::operator[]: index out of range.");
        }
        T res;

        dpct::get_in_order_queue().memcpy(&res, &this->get_data_ptr()[idx], sizeof(T)).wait();
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
        device_no_old = dpct::get_current_device_id();

        if (device_ != device_no_old) {
            /*
            DPCT1093:20: The "device_" device may be not the one intended for use. Adjust the selected device if needed.
            */
            dpct::select_device(device_);
        }

        try {
            (this->data_) = (typename std::remove_reference<decltype(this->data_)>::type)
                                 sycl::malloc_device(size, dpct::get_in_order_queue());
        } catch ( ... ) {
            size_t free = 0, total = 0;
            /*
            DPCT1106:21: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be
            supported by all compilers or runtimes. You may need to adjust the code.
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
            DPCT1093:22: The "device_no_old" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            dpct::select_device(device_no_old);
        }
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      throw;
    }

    template <typename T> 
    void cuNDArray<T>::deallocate_memory()
    {
        if (this->data_) {

            int device_no_old;

            device_no_old = dpct::get_current_device_id();
            if (device_ != device_no_old) {
                /*
                DPCT1093:23: The "device_" device may be not the one intended for use. Adjust the selected device if
                needed.
                */

                dpct::select_device(device_);
            }


            dpct::dpct_free(this->data_, dpct::get_in_order_queue());
            if (device_ != device_no_old) {
                /*
                DPCT1093:24: The "device_no_old" device may be not the one intended for use. Adjust the selected device
                if needed.
                */

                dpct::select_device(device_no_old);
            }
            this->data_ = 0;
        }
    }
}
