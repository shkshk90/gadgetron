#pragma once
#include <mutex>

// #include <boost/process.hpp>
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/system.hpp>

namespace Gadgetron::Process {

    namespace detail{
        std::lock_guard<std::mutex> obtain_process_lock();
    }

    template<class ... ARGS>
    boost::process::child child(ARGS&&... args){
        auto guard = detail::obtain_process_lock();
        return boost::process::child(std::forward<ARGS>(args)...);
    }

    template<class... ARGS>
    auto system(ARGS&&... args){
        auto guard = detail::obtain_process_lock();
        return boost::process::system(std::forward<ARGS>(args)...);
    }

}
