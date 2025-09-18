#pragma once

#ifndef DPCT_PROFILING_ENABLED
#define DPCT_PROFILING_ENABLED
#endif 

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <string>

namespace Gadgetron{

    class GPUTimer
    {
    public:
        GPUTimer() : name_("GPUTimer"), timing_in_destruction_(true)
        {
            start();
        }

        GPUTimer(bool timing) : name_("GPUTimer"), timing_in_destruction_(timing)
        {
            if ( timing_in_destruction_ )
            {
                start();
            }
        }

        GPUTimer(const char* name) : name_(name), timing_in_destruction_(true)
        {
            start();
        }

        virtual ~GPUTimer() 
        {
            if ( timing_in_destruction_ )
            {
                stop();
            }
        }

        virtual void start()
        {
            dpct::device_ext& dev_ct1 = dpct::get_current_device();
            sycl::queue& q_ct1 = dev_ct1.in_order_queue();
            start_event_ = new sycl::event();
            stop_event_ = new sycl::event();
            dpct::sync_barrier(start_event_, &q_ct1);
        }

        virtual void stop()
        {
            dpct::device_ext& dev_ct1 = dpct::get_current_device();
            sycl::queue& q_ct1 = dev_ct1.in_order_queue();
            float time;
            dpct::sync_barrier(stop_event_, &q_ct1);
            stop_event_->wait_and_throw();
            time = (stop_event_->get_profiling_info<sycl::info::event_profiling::command_end>() -
                    start_event_->get_profiling_info<sycl::info::event_profiling::command_start>()) /
                   1000000.0f;
            dpct::destroy_event(start_event_);
            dpct::destroy_event(stop_event_);

            // GDEBUG_STREAM(name_ << ": " << time << " ms" << std::endl);
        }

        void set_timing_in_destruction(bool timing) { timing_in_destruction_ = timing; }

        dpct::event_ptr start_event_;
        dpct::event_ptr stop_event_;

        std::string name_;
        bool timing_in_destruction_;
    };
}