////////////////////////////////////////////////////////////////////////////
// complext.h
////////////////////////////////////////////////////////////////////////////

/** \file complext.h
    \brief An implementation of complex numbers that works for both the cpu and gpu.

    complext.h provides an implementation of complex numbers that, unlike std::complex,
    works on both the cpu and gpu. 
    It follows the interface defined for std::complex.
*/

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <complex>
#include <cmath>
#include <iostream>


// complext
#include <algorithm>
#include <cstdint> // for size_t
#include <type_traits>
#include <memory>
#include <cstring>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <new>
#include <vector>
#include <array>
#include <numeric>
#include <sstream>
#include <dpct/dpl_utils.hpp>

#include <float.h>


////////////////////////////////////////////////////////////////////////////
// log.h
////////////////////////////////////////////////////////////////////////////



#include <mutex>

#define GADGETRON_LOG_MASK_ENVIRONMENT "GADGETRON_LOG_MASK"
#define GADGETRON_LOG_FILE_ENVIRONMENT "GADGETRON_LOG_FILE"
#define BOOST_THROW_EXCEPTION(tt) throw std::tt;

namespace Gadgetron
{
  /**
     Gadgetron log levels
   */
  enum GadgetronLogLevel
  {
    GADGETRON_LOG_LEVEL_DEBUG = 0,     //!< Debug information
    GADGETRON_LOG_LEVEL_INFO,          //!< Regular application information
    GADGETRON_LOG_LEVEL_WARNING,       //!< Warnings about events that could lead to failues
    GADGETRON_LOG_LEVEL_ERROR,         //!< Errors after which application will be unable to continue
    GADGETRON_LOG_LEVEL_VERBOSE,       //!< Verbose information about algorithm parameters, etc. 
    GADGETRON_LOG_LEVEL_MAX            //!< All log levels must have values lower than this
  };

  /**
     Gadgetron output options. These options control what context information
     will be printed with the log statements.
   */
  enum GadgetronLogOutput
  {
    GADGETRON_LOG_PRINT_FILELOC = 0,  //!< Print filename and line in file
    GADGETRON_LOG_PRINT_FOLDER,       //!< Print the folder name too (full filename)
    GADGETRON_LOG_PRINT_LEVEL,        //!< Print Log Level
    GADGETRON_LOG_PRINT_DATETIME,     //!< Print date and time
    GADGETRON_LOG_PRINT_MAX           //!< All print options must have lower values than this
  };

  class GadgetronLogger
  {
  public:
    ///Function for accessing the process wide singleton
    static GadgetronLogger* instance();

    ///Generic log function. Use the logging macros for easy access to this function
    void log(GadgetronLogLevel LEVEL, const char* filename, int lineno, const char* cformatting, ...);

    void enableLogLevel(GadgetronLogLevel LEVEL);
    void disableLogLevel(GadgetronLogLevel LEVEL);
    bool isLevelEnabled(GadgetronLogLevel LEVEL);
    void enableAllLogLevels();
    void disableAllLogLevels();

    void enableOutputOption(GadgetronLogOutput OUTPUT);
    void disableOutputOption(GadgetronLogOutput OUTPUT);
    bool isOutputOptionEnabled(GadgetronLogOutput OUTPUT);
    void enableAllOutputOptions();
    void disableAllOutputOptions();

  protected:
    GadgetronLogger();
    static GadgetronLogger* instance_;
    std::vector<bool> level_mask_;
    std::vector<bool> print_mask_;
    std::mutex m;
  };
}

#define GDEBUG(...)   Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_DEBUG,   __FILE__, __LINE__, __VA_ARGS__)
#define GINFO(...)    Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_INFO,    __FILE__, __LINE__, __VA_ARGS__)
#define GWARN(...)    Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define GERROR(...)   Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_ERROR,   __FILE__, __LINE__, __VA_ARGS__)
#define GVERBOSE(...) Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_VERBOSE,   __FILE__, __LINE__, __VA_ARGS__)

#define GEXCEPTION(err, message);	  \
  {					  \
    std::string gdb(message);		  \
    gdb += std::string(" --> ");	  \
    gdb += err.what();			  \
    GDEBUG(gdb.c_str());		  \
 }

//Stream syntax log level functions
#define GINFO_STREAM(message)				\
  {							\
    std::stringstream gadget_msg_dep_str;		\
    gadget_msg_dep_str  << message << std::endl;	\
    GINFO(gadget_msg_dep_str.str().c_str());		\
  }

#define GVERBOSE_STREAM(message)					\
  {								\
    std::stringstream gadget_msg_dep_str;			\
    gadget_msg_dep_str  << message << std::endl;		\
    GVERBOSE(gadget_msg_dep_str.str().c_str());			\
  }

#ifndef MATLAB_MEX_COMPILE

#define GDEBUG_STREAM(message)				\
{							\
    std::stringstream gadget_msg_dep_str;		\
    gadget_msg_dep_str  << message << std::endl;	\
    GDEBUG(gadget_msg_dep_str.str().c_str());		\
}

#define GWARN_STREAM(message)					\
  {								\
    std::stringstream gadget_msg_dep_str;			\
    gadget_msg_dep_str  << message << std::endl;		\
    GWARN(gadget_msg_dep_str.str().c_str());			\
  }

#define GERROR_STREAM(message)					\
  {								\
    std::stringstream gadget_msg_dep_str;			\
    gadget_msg_dep_str  << message << std::endl;		\
    GERROR(gadget_msg_dep_str.str().c_str());			\
  }

#else
    #pragma message ("Use matlab definition for GDEBUG stream ... ")

    #ifdef _DEBUG
        #define GDEBUG_STREAM(message) { std::ostringstream outs; outs << " (" << __FILE__ << ", " << __LINE__ << "): " << message << std::endl << '\0'; mexPrintf("%s", outs.str().c_str()); }
    #else
        #define GDEBUG_STREAM(message) { std::ostringstream outs; outs << message << std::endl << '\0'; mexPrintf("%s", outs.str().c_str()); }
    #endif // _DEBUG

    #ifdef _DEBUG
        #define GWARN_STREAM(message) { std::ostringstream outs; outs << " (" << __FILE__ << ", " << __LINE__ << "): " << message << std::endl << '\0'; mexWarnMsgTxt(outs.str().c_str()); }
    #else
        #define GWARN_STREAM(message) { std::ostringstream outs; outs << message << std::endl << '\0'; mexWarnMsgTxt(outs.str().c_str()); }
    #endif // _DEBUG

    #define GERROR_STREAM(message) GDEBUG_STREAM(message) 
#endif // MATLAB_MEX_COMPILE

//Older debugging macros
//TODO: Review and check that they are up to date
#define GDEBUG_CONDITION_STREAM(con, message) { if ( con ) GDEBUG_STREAM(message) }
#define GWARN_CONDITION_STREAM(con, message) { if ( con ) GWARN_STREAM(message) }
     
#define GADGET_THROW(msg) { GERROR_STREAM(msg); throw std::runtime_error(msg); }
#define GADGET_CHECK_THROW(con) { if ( !(con) ) { GERROR_STREAM(#con); throw std::runtime_error(#con); } }

#define GADGET_CATCH_THROW(con) { try { con; } catch(...) { GERROR_STREAM(#con); throw std::runtime_error(#con); } }

#define GADGET_CHECK_RETURN(con, value) { if ( ! (con) ) { GERROR_STREAM("Returning '" << value << "' due to failed check: '" << #con << "'"); return (value); } }
#define GADGET_CHECK_RETURN_FALSE(con) { if ( ! (con) ) { GERROR_STREAM("Returning false due to failed check: '" << #con << "'"); return false; } }

#define GADGET_CHECK_EXCEPTION_RETURN(con, value) { try { con; } catch(...) { GERROR_STREAM("Returning '" << value << "' due to failed check: '" << #con << "'"); return (value); } }
#define GADGET_CHECK_EXCEPTION_RETURN_FALSE(con) { try { con; } catch(...) { GERROR_STREAM("Returning false due to failed check: '" << #con << "'"); return false; } }

#ifdef GADGET_DEBUG_MODE
#define GADGET_DEBUG_CHECK_THROW(con) GADGET_CHECK_THROW(con)
#define GADGET_DEBUG_CHECK_RETURN(con, value) GADGET_CHECK_RETURN(con, value)
#define GADGET_DEBUG_CHECK_RETURN_FALSE(con) GADGET_CHECK_RETURN_FALSE(con)
#else
#define GADGET_DEBUG_CHECK_THROW(con) 
#define GADGET_DEBUG_CHECK_RETURN(con, value) 
#define GADGET_DEBUG_CHECK_RETURN_FALSE(con) 
#endif // GADGET_DEBUG_MODE






////////////////////////////////////////////////////////////////////////////
// real_utilities.h
////////////////////////////////////////////////////////////////////////////

namespace Gadgetron{
  
  class cuda_error : public std::runtime_error
  {
  public:
    cuda_error(std::string msg) : std::runtime_error(msg) {}
    /*
    DPCT1009:0: SYCL reports errors using exceptions and does not use error codes. Please replace the
    "get_error_string_dummy(...)" with a real error-handling function.
    */
    cuda_error(dpct::err0 errN) : std::runtime_error(dpct::get_error_string_dummy(errN)) {
    }
  };
}

namespace Gadgetron {

  /**
   *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
   *  inspired by cutil.h: CUT_CHECK_ERROR
   */
  inline void CHECK_FOR_CUDA_ERROR(char const * cur_fun, const char* file, const int line) {
    /*
    DPCT1010:3: SYCL uses exceptions to report errors and does not use the error codes. The cudaGetLastError function
    call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 errorCode = 0;
    /*
    DPCT1000:2: Error handling if-stmt was detected but could not be rewritten.
    */
    if (errorCode != 0) {
      /*
      DPCT1001:1: The statement could not be removed.
      */
      throw cuda_error(errorCode);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess) {
      throw cuda_error(errorCode);
    }
#endif
  }
}

/**
 *  Checks for CUDA errors and throws an exception if an error was detected.
 */
#define CHECK_FOR_CUDA_ERROR(); CHECK_FOR_CUDA_ERROR(__func__,__FILE__,__LINE__);

/**
 *  Call "res", checks for CUDA errors and throws an exception if an error was detected.
 */
/*
DPCT1001:4: The statement could not be removed.
*/
/*
DPCT1000:5: Error handling if-stmt was detected but could not be rewritten.
*/
#define CUDA_CALL(res) { dpct::err0 errorCode = res; if (errorCode != 0) { throw cuda_error(errorCode); } }

#define CUSPARSE_CALL(res) {cusparseStatus_t errorCode = res; \
    if (errorCode != CUSPARSE_STATUS_SUCCESS){ \
        std::stringstream ss; \
        ss << "CUSPARSE failed with error: " <<  gadgetron_getCusparseErrorString(errorCode); \
        throw cuda_error(ss.str());}}

////////////////////////////////////////////////////////////////////////////
// real_utilities.h
////////////////////////////////////////////////////////////////////////////

#ifdef _USE_MATH_DEFINES
#include <math.h>
#else
#define _USE_MATH_DEFINES
#include <math.h>
#undef _USE_MATH_DEFINES
#endif

#include <float.h>

//
// Get scalar limits of operation
//

template<class T> __inline__ T get_min();
template<class T> __inline__ T get_max();
template<class T> __inline__ T get_epsilon();

//
// Math prototypes
//

template<class REAL> __inline__ REAL get_pi();

//
// Implementation
//

template<> __inline__ float get_min<float>()
{
  return FLT_MIN;
}

template<> __inline__ double get_min<double>()
{
  return DBL_MIN;
}

template<> __inline__ float get_max<float>()
{
  return FLT_MAX;
}

template<> __inline__ double get_max<double>()
{
  return DBL_MAX;
}

template<> __inline__ float get_epsilon<float>()
{
  return FLT_EPSILON;
}

template<> __inline__ double get_epsilon<double>()
{
  return DBL_EPSILON;
}

template<> __inline__ float get_pi() { return (float)M_PI; }
template<> __inline__ double get_pi() { return M_PI; }


////////////////////////////////////////////////////////////////////////////
// Types.h
////////////////////////////////////////////////////////////////////////////

namespace Gadgetron { namespace Core {

    // Adapted from cppreference.com

    namespace { namespace gadgetron_types_detail {
        template <class F, class Tuple, std::size_t... I>
        inline constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
            return f(get<I>(std::forward<Tuple>(t))...);
        }
    }}

    template <class F, class TupleLike> constexpr decltype(auto) apply(F&& f, TupleLike&& t) {
        return gadgetron_types_detail::apply_impl(std::forward<F>(f), std::forward<TupleLike>(t),
            std::make_index_sequence<std::tuple_size<std::remove_reference_t<TupleLike>>::value>{});
    }
    

}}

////////////////////////////////////////////////////////////////////////////
// typetraits.h
////////////////////////////////////////////////////////////////////////////

namespace Gadgetron { namespace Core {

    template <class T> constexpr bool is_trivially_copyable_v = std::is_trivially_copyable<T>::value;

    template <class T, class V> constexpr bool is_same_v = std::is_same<T, V>::value;

    template <class T, class V> constexpr bool is_convertible_v = std::is_convertible<T, V>::value;

    template <bool... ARGS> struct all_of;

    template <> struct all_of<> : std::true_type {};

    template <bool... ARGS> struct all_of<false, ARGS...> : std::false_type {};

    template <bool... ARGS> struct all_of<true, ARGS...> : all_of<ARGS...> {};


    template <bool... ARGS> constexpr bool all_of_v = all_of<ARGS...>::value;


    template <bool... ARGS> struct any_of;
    template <> struct any_of<> : std::false_type{};

    template <bool... ARGS> struct any_of<true, ARGS...> : std::true_type {};

    template <bool... ARGS> struct any_of<false, ARGS...> : any_of<ARGS...> {};

    template <bool... ARGS> constexpr bool any_of_v = any_of<ARGS...>::value;

    template<class T>
    constexpr bool is_floating_point_v = std::is_floating_point<T>::value;
}}

////////////////////////////////////////////////////////////////////////////
// complext.h
////////////////////////////////////////////////////////////////////////////

namespace Gadgetron {

    using std::abs; // workaround for nvcc
    using std::sin;
    using std::cos;
    using std::exp;
    using std::sqrt;
    using std::atan2;

    /**
     * \class complext
     * \brief An implementation of complex numbers that works for both the cpu and gpu.
     */
    template<class T>
    class complext {
    public:

//    T vec[2];
        T _real;
        T _imag;

        __inline__ T real() const {
            return _real;
        }

        __inline__ T imag() const {
            return _imag;
        }

        __inline__ complext() = default;

        __inline__ complext(T real, T imag) {
            _real = real;
            _imag = imag;
        }

        template<class R>
        __inline__ complext(const complext<R> &tmp) {
            _real = tmp._real;
            _imag = tmp._imag;
        }

        __inline__ complext(const std::complex<T> &tmp) {
            _real = tmp.real();
            _imag = tmp.imag();
        }

        template<class R>
        __inline__ complext(const std::complex<R> &tmp) {
            _real = tmp.real();
            _imag = tmp.imag();
        }

        __inline__ complext(const T r) {
            _real = r;
            _imag = T(0);
        }

        __inline__ void conj() {
            _imag = -_imag;
        }
/*
        __inline__ __host__ __device__  complext<T> operator+(const complext<T> &other) {
            return complext<T>(_real + other._real, _imag + other._imag);
        }

        __inline__ __host__ __device__  complext<T> operator-(const complext<T> &other) {
            return complext<T>(_real - other._real, _imag - other._imag);
        }
*/
        __inline__ complext<T> operator-() {
            return complext<T>(-_real, -_imag);
        }

        __inline__ void operator-=(const complext<T> &other) {
            _real -= other._real;
            _imag -= other._imag;
        }

        __inline__ void operator+=(const complext<T> &other) {
            _real += other._real;
            _imag += other._imag;
        }

        __inline__ complext<T> operator*(const T &other) {
            return complext<T>(_real * other, _imag * other);
        }
/*
        __inline__ __host__ __device__  complext<T> operator*(const complext<T> &other) {
            return complext<T>(_real * other._real - _imag * other._imag,
                               _real * other._imag + _imag * other._real);
        }
*/
        __inline__ complext<T> operator/(const T &other) {
            return complext<T>(_real / other, _imag / other);
        }
/*
        __inline__ __host__ __device__  complext<T> operator/(const complext<T> &other) {
            T cd = other._real * other._real + other._imag * other._imag;
            return complext<T>((_real * other._real + _imag * other._imag) / cd,
                               (_imag * other._real - _real * other._imag) / cd);
        }
*/

        __inline__ void operator*=(const T &other) {
            _real *= other;
            _imag *= other;
        }

        __inline__ void operator*=(const complext<T> &other) {
            complext<T> tmp = *this;
            _real = tmp._real * other._real - tmp._imag * other._imag;
            _imag = tmp._real * other._imag + tmp._imag * other._real;
        }

        __inline__ void operator/=(const T &other) {
            _real /= other;
            _imag /= other;
        }

        __inline__ void operator/=(const complext<T> &other) {
            complext<T> tmp = (*this) / other;
            _real = tmp._real;
            _imag = tmp._imag;
        }

        __inline__ bool operator==(const complext<T> &comp2) {

            return _real == comp2._real && _imag == comp2._imag;
        }

        __inline__ bool operator!=(const complext<T> &comp2) {

            return not(*this == comp2);
        }
    };

    template<typename T>
    inline std::ostream &operator<<(std::ostream &os, const complext<T> &a) {
        os << a.real() << (a.imag() > 0 ? "+" : " " )  << a.imag() << "i";
        return os;
    }


    typedef complext<float> float_complext;
    typedef complext<double> double_complext;

    template <class T> struct is_complex_type { static constexpr bool value = false; };
    template <class T> struct is_complex_type<std::complex<T>> { static constexpr bool value = true; };
    template <class T> struct is_complex_type<complext<T>> { static constexpr bool value = true; };
    template<class T> constexpr bool is_complex_type_v = is_complex_type<T>::value;

    template<class T>
    struct realType {
    };
    template<>
    struct realType<short> {
        typedef double Type;
    };
    template<>
    struct realType<unsigned short> {
        typedef double Type;
    };
    template<>
    struct realType<int> {
        typedef double Type;
    };
    template<>
    struct realType<unsigned int> {
        typedef double Type;
    };
    template<>
    struct realType<float_complext> {
        typedef float Type;
    };
    template<>
    struct realType<double_complext> {
        typedef double Type;
    };
    template<>
    struct realType<float> {
        typedef float Type;
    };
    template<>
    struct realType<double> {
        typedef double Type;
    };
    template<>
    struct realType<std::complex<float> > {
        typedef float Type;
    };
    template<>
    struct realType<std::complex<double> > {
        typedef double Type;
    };

    template<class T>
    using realType_t = typename realType<T>::Type;

    template<class T>
    struct stdType {
        typedef T Type;
    };
    template<>
    struct stdType<double_complext> {
        typedef std::complex<double> Type;
    };
    template<>
    struct stdType<float_complext> {
        typedef std::complex<float> Type;
    };
    template<>
    struct stdType<std::complex<double> > {
        typedef std::complex<double> Type;
    };
    template<>
    struct stdType<std::complex<float> > {
        typedef std::complex<float> Type;
    };
    template<>
    struct stdType<double> {
        typedef double Type;
    };
    template<>
    struct stdType<float> {
        typedef float Type;
    };



    __inline__ double sgn(double x) {
        return (double(0) < x) - (x < double(0));
    }

    __inline__ float sgn(float x) {
        return (float) ((float(0) < x) - (x < float(0)));
    }

    template<class T>
    __inline__ complext<T> sgn(complext<T> x) {
        if (norm(x) <= T(0)) return complext<T>(0);
        return (x / abs(x));
    }

    template<class T>
    __inline__ complext<T> polar(const T &rho, const T &theta = 0) {
        return complext<T>(rho * std::cos(theta), rho * std::sin(theta));
    }

    template<class T>
    __inline__ complext<T> sqrt(complext<T> x) {
        T r = abs(x);
        return complext<T>(::sqrt((r + x.real()) / 2), sgn(x.imag()) * ::sqrt((r - x.real()) / 2));
    }

    template<class T>
    __inline__ T abs(complext<T> comp) {
        return sqrt(comp._real * comp._real + comp._imag * comp._imag);
    }

    template<class T>
    __inline__ complext<T> sin(complext<T> comp) {
        return complext<T>(sin(comp._real) * std::cosh(comp._imag), std::cos(comp._real) * std::sinh(comp._imag));
    }

    template<class T>
    __inline__ complext<T> cos(complext<T> comp) {
        return complext<T>(cos(comp._real) * cosh(comp._imag), -sin(comp._real) * sinh(comp._imag));
    }

    template<class T>
    __inline__ complext<T> exp(complext<T> com) {
        return exp(com._real) * complext<T>(cos(com._imag), sin(com._imag));
    }

    template<class T>
    __inline__ T imag(complext<T> comp) {
        return comp._imag;
    }

    __inline__ double real(double r) {
        return r;
    }

    __inline__ double imag(double r) {
        return 0.0;
    }

    __inline__ float real(float r) {
        return r;
    }

    __inline__ float imag(float r) {
        return 0.0f;
    }

    template<class T>
    __inline__ T real(complext<T> comp) {
        return comp._real;
    }

    template<class T>
    __inline__ T arg(complext<T> comp) {
        return atan2(comp._imag, comp._real);
    }

    template<class T, class S>
    __inline__ auto operator*(const complext<T>& c1, const S& c2) -> complext<decltype(c1._real*c2)>{
        return c2*c1;
    };

    template<class T, class S>
    __inline__ auto operator*(const T& c1, const complext<S>& c2) -> complext<decltype(c1*c2._real)>{
        auto real = c1*c2._real;
        auto imag = c1*c2._imag;

        return complext<decltype(real)>(real,imag);
    };


    template<class T, class S>
    __inline__ auto operator+(const T& c1, const complext<S>& c2) -> complext<decltype(c1+c2._real)>{
        auto real = c1+c2._real;
        return complext<decltype(real)>(real,c2._imag);
    };


    template<class T, class S>
    __inline__ auto operator+(const complext<T>& c1, const S& c2) -> complext<decltype(c1._real+c2)>{
        return c2+c1;
    };


    template<class T, class S>
    __inline__ auto operator/(const complext<T>& c1, const S& c2) -> complext<decltype(c1._real*c2)>{
        auto real = c1._real / c2;
        auto imag = c1._imag / c2;
        return complext<decltype(real)>(real,imag);
    };

    template<class T, class S>
    __inline__ auto operator/(const T& c1, const complext<S>& c2)-> complext<decltype(c1/c2._real)>{

        auto real = c1*c2._real;
        auto imag = -c2._imag*c1;
        auto denum = c2._real*c2._real+c2._imag*c2._imag;

        return complext<decltype(real)>(real/denum,imag/denum);
    };

    template<class T, class S>
    __inline__ auto operator-(const T& c1, const complext<S>& c2) -> complext<decltype(c1-c2._real)>{
        auto real = c1-c2._real;
        return complext<decltype(real)>(real,c2._imag);
    };

    template<class T, class S>
    __inline__ auto operator-(const complext<T>& c1, const S& c2) -> complext<decltype(c1._real-c2)>{
        auto real = c1._real-c2;
        return complext<decltype(real)>(real,c1._imag);
    };



    template<class T, class S>
    __inline__ auto operator*(const complext<T>& c1, const complext<S>& c2) -> complext<decltype(c1._real*c2._real)>{
        auto real = c1._real*c2._real-c1._imag*c2._imag;
        auto imag = c1._imag*c2._real+c1._real*c2._imag;

        return complext<decltype(real)>(real,imag);
    };

    template<class T, class S>
    __inline__ auto operator+(const complext<T>& c1, const complext<S>& c2)-> complext<decltype(c1._real+c2._real)>{
        auto real = c1._real+c2._real;
        auto imag = c1._imag+c2._imag;

        return complext<decltype(real)>(real,imag);
    };
    template<class T, class S>
    __inline__ auto operator-(const complext<T>& c1, const complext<S>& c2)-> complext<decltype(c1._real-c2._real)>{
        auto real = c1._real-c2._real;
        auto imag = c1._imag-c2._imag;

        return complext<decltype(real)>(real,imag);
    };
    template<class T, class S>
    __inline__ auto operator/(const complext<T>& c1, const complext<S>& c2)-> complext<decltype(c1._real/c2._real)>{

        auto real = c1._real*c2._real+c1._imag * c2._imag;
        auto imag = c2._real*c1._imag-c2._imag*c1._real;
        auto denum = c2._real*c2._real+c2._imag*c2._imag;

        return complext<decltype(real)>(real/denum,imag/denum);
    };



    __inline__ float norm(const float &r) {
        return r * r;
    }

    __inline__ double norm(const double &r) {
        return r * r;
    }

    template<class T>
    __inline__ T norm(const complext<T> &z) {
        return z._real * z._real + z._imag * z._imag;
    }

    __inline__ double conj(const double &r) {
        return r;
    }

    __inline__ float conj(const float &r) {
        return r;
    }

    template<class T>
    __inline__ complext<T> conj(const complext<T> &z) {
        complext<T> res = z;
        res.conj();
        return res;
    }
}

////////////////////////////////////////////////////////////////////////////
// vector_td.h
////////////////////////////////////////////////////////////////////////////

/** \file vector_td.h
    \brief The class vector_td defines a D-dimensional vector of type T.

    The class vector_td defines a D-dimensional vector of type T.
    It is used in the Gadgetron to represent short vectors.
    I.e. it is purposedly templetated with dimensionality D as type unsigned int instead of size_t.
    For larger vectors consider using the NDArray class instead (or a std::vector).
    The vector_td class can be used on both the cpu and gpu.
    The accompanying headers vector_td_opeators.h and vector_td_utilities.h define most of the functionality.
    Note that vector_td should not be used to represent complex numbers. For that we provide the custom class complext
   instead.
*/



#ifdef max
#undef max
#endif // max

namespace Gadgetron {

    template <class T, unsigned int D> class vector_td {
    public:
        T vec[D];
        __inline__ vector_td() = default;

        template <typename... X, typename = std::enable_if_t<(sizeof...(X) > 1)>> constexpr __inline__ explicit vector_td(X... xs) : vec{ T(xs)... } {}

        // __inline__ vector_td(const vector_td& other) = default;

        template <class T2> __inline__ explicit vector_td(const vector_td<T2, D>& other) {
            for (unsigned int i = 0; i < D; i++)
                vec[i] = (T)other[i];
        }

        __inline__ explicit vector_td(T x) {
            for (unsigned int i = 0; i < D; i++)
                vec[i] = x;
        }

        template <class STATIC_CONTAINER, class SFINAE = std::enable_if_t<STATIC_CONTAINER().size() == D>>
        explicit vector_td(const STATIC_CONTAINER& other) {
            std::copy(other.begin(), other.end(), this->begin());
        }

        template <class TI, typename std::enable_if<(D > 1) && std::is_convertible<TI,T>::value >::type* = nullptr>
        explicit vector_td(TI input[D]) {
            std::copy(input,input+D,vec);

        }
        __inline__ T& operator[](size_t i) {
            return vec[i];
        }

        __inline__ const T& operator[](size_t i) const {
            return vec[i];
        }

        __inline__ T* begin() {
            return vec;
        }
        __inline__ const T* begin() const {
            return vec;
        }
        __inline__ T* end() {
            return vec + D;
        }
        __inline__ const T* end() const {
            return vec + D;
        }

        static constexpr size_t size() {
            return D;
        }
    };

    template <class T, class... ARGS> auto make_vector_td(ARGS&&... args) {
        return vector_td<T, sizeof...(args)>{ std::forward<ARGS>(args)... };
    }

    //
    // Some typedefs for convenience (templated typedefs are not (yet) available in C++)
    //

    template <class REAL, unsigned int D> struct reald { typedef vector_td<REAL, D> Type; };

    template <unsigned int D> struct uintd { typedef vector_td<unsigned int, D> Type; };

    template <unsigned int D> struct uint64d { typedef vector_td<size_t, D> Type; };

    template <unsigned int D> struct intd { typedef vector_td<int, D> Type; };

    template <unsigned int D> struct int64d { typedef vector_td<long long, D> Type; };

    template <unsigned int D> struct floatd { typedef typename reald<float, D>::Type Type; };

    template <unsigned int D> struct doubled { typedef typename reald<double, D>::Type Type; };


    typedef vector_td<unsigned int, 1> uintd1;
    typedef vector_td<unsigned int, 2> uintd2;
    typedef vector_td<unsigned int, 3> uintd3;
    typedef vector_td<unsigned int, 4> uintd4;
    typedef vector_td<unsigned int, 5> uintd5;

    typedef vector_td<size_t, 1> uint64d1;
    typedef vector_td<size_t, 2> uint64d2;
    typedef vector_td<size_t, 3> uint64d3;
    typedef vector_td<size_t, 4> uint64d4;
    typedef vector_td<size_t, 5> uint64d5;

    typedef vector_td<int, 1> intd1;
    typedef vector_td<int, 2> intd2;
    typedef vector_td<int, 3> intd3;
    typedef vector_td<int, 4> intd4;
    typedef vector_td<int, 5> intd5;

    typedef vector_td<long long, 1> int64d1;
    typedef vector_td<long long, 2> int64d2;
    typedef vector_td<long long, 3> int64d3;
    typedef vector_td<long long, 4> int64d4;
    typedef vector_td<long long, 5> int64d5;

    typedef vector_td<float, 1> floatd1;
    typedef vector_td<float, 2> floatd2;
    typedef vector_td<float, 3> floatd3;
    typedef vector_td<float, 4> floatd4;
    typedef vector_td<float, 5> floatd5;

    typedef vector_td<double, 1> doubled1;
    typedef vector_td<double, 2> doubled2;
    typedef vector_td<double, 3> doubled3;
    typedef vector_td<double, 4> doubled4;
    typedef vector_td<double, 5> doubled5;
}

template <class T, unsigned int N> class std::tuple_size<Gadgetron::vector_td<T, N>> : public std::integral_constant<size_t, N> {};

template <std::size_t I, class T, unsigned int N> struct std::tuple_element<I, Gadgetron::vector_td<T, N>> {
    using type = T;
};

template <size_t I, class T, unsigned int N>
    constexpr std::enable_if_t < I<N, T&> get(Gadgetron::vector_td<T, N>& a) noexcept {
    return a[I];
}

template <size_t I, class T, unsigned int N>
    constexpr std::enable_if_t < I<N, const T&> get(const Gadgetron::vector_td<T, N>& a) noexcept {
    return a[I];
}

namespace Gadgetron {
    template <size_t I, class T, unsigned int N>
        constexpr std::enable_if_t < I<N, T&> get(Gadgetron::vector_td<T, N>& a) noexcept {
        return a[I];
    }
    template <size_t I, class T, unsigned int N>
        constexpr std::enable_if_t < I<N, const T&> get(const Gadgetron::vector_td<T, N>& a) noexcept {
        return a[I];
    }
}


namespace Gadgetron{

  //
  // Return types
  //

  template <class T, class I> struct vectorTDReturnType {};
  template <class T> struct vectorTDReturnType<T,T> {typedef T type;};
  template<> struct vectorTDReturnType<unsigned int, int> {typedef int type;};
  template<> struct vectorTDReturnType<int, unsigned int> {typedef int type;};
  template<> struct vectorTDReturnType<int, bool> {typedef int type;};
  template<> struct vectorTDReturnType<bool,int> {typedef int type;};
  template<> struct vectorTDReturnType<unsigned int, bool> {typedef int type;};
  template<> struct vectorTDReturnType<bool,unsigned int> {typedef int type;};
  template<> struct vectorTDReturnType<float, unsigned int> {typedef float type;};
  template<> struct vectorTDReturnType<unsigned int, float> {typedef float type;};
  template<> struct vectorTDReturnType<float, int> {typedef float type;};
  template<> struct vectorTDReturnType<int, float> {typedef float type;};
  template<> struct vectorTDReturnType<float, bool> {typedef float type;};
	template<> struct vectorTDReturnType<bool, float> {typedef float type;};
  template<> struct vectorTDReturnType<double, unsigned int> {typedef double type;};
  template<> struct vectorTDReturnType<unsigned int, double> {typedef double type;};
  template<> struct vectorTDReturnType<double, int> {typedef double type;};
  template<> struct vectorTDReturnType<int, double> {typedef double type;};
  template<> struct vectorTDReturnType<double, bool> {typedef double type;};
  template<> struct vectorTDReturnType<bool, double> {typedef double type;};
  template<> struct vectorTDReturnType<double, float> {typedef double type;};
  template<> struct vectorTDReturnType<float,double> {typedef double type;};

  //
  // Operators are defined as component wise operations.
  //

  //
  // Arithmetic operators
  //

  template< class T,class R,  unsigned int D > __inline__ 
  void operator+= ( vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] += v2.vec[i];
  }

  template< class T,class R,  unsigned int D > __inline__ 
  void operator+= ( vector_td<T,D> &v1, const R &v2 )
  {
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] += v2;
  }

  template< class T,class R,  unsigned int D > __inline__ 
  void operator-= ( vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] -= v2.vec[i];
  }


  template< class T, class R, unsigned int D > __inline__ 
  void operator*= ( vector_td<T,D> &v1, const R &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] *= v2;
  }

  template< class T,class R,  unsigned int D > __inline__ 
  void operator *=  ( vector_td<T,D> &v1, const vector_td<R,D> &v2 )
	{
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] *= v2.vec[i];
  }

  template< class T,class R,  unsigned int D > __inline__ 
  void operator /= ( vector_td<T,D> &v1, const R &v2 )
  {
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] /= v2;
  }

  template< class T,class R,  unsigned int D > __inline__ 
  void operator /=  ( vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  {
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] /= v2.vec[i];
  }

  template< class T,class R,  unsigned int D > __inline__ 
  void component_wise_div_eq ( vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] /= v2.vec[i];
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator+ ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    vector_td<typename vectorTDReturnType<T,R>::type,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = v1.vec[i]+v2.vec[i];
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator+ ( const vector_td<T,D> &v1, const R &v2 )
  {
    vector_td<typename vectorTDReturnType<T,R>::type,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = v1.vec[i]+v2;
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator- ( const vector_td<T,D> &v1, const R &v2 )
  {
    vector_td<typename vectorTDReturnType<T,R>::type,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = v1.vec[i]-v2;
    return res;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator+ (const R &v2, const vector_td<T,D> &v1 )
  {
    return v1+v2;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator- ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    vector_td<typename vectorTDReturnType<T,R>::type,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = v1.vec[i]-v2.vec[i];
    return res;
  }

  template< class T, unsigned int D > __inline__ 
  vector_td<T,D> operator- ( const vector_td<T,D> &v1)
  {
    vector_td<T,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = -v1.vec[i];
    return res;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> component_wise_mul ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    vector_td<typename vectorTDReturnType<T,R>::type,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = v1.vec[i]*v2.vec[i];
    return res;
  }

  template< class T, unsigned int D > __inline__ 
  vector_td<T,D> component_wise_mul ( const vector_td<T,D> &v1, const vector_td<T,D> &v2 )
  {
    vector_td<T,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = v1.vec[i]*v2.vec[i];
    return res;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator* ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  {
    vector_td<typename vectorTDReturnType<T,R>::type,D> res;
    for(unsigned int i=0; i<D; i++ )  res.vec[i] = v1.vec[i]*v2.vec[i];
    return res;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator* ( const vector_td<T,D> &v1, const R &v2 )
  { 
    vector_td<typename vectorTDReturnType<T,R>::type,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = v1.vec[i]*v2;
    return res;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator* ( const R &v1, const vector_td<T,D> &v2 )
  { 
    return v2*v1;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator/ ( const vector_td<T,D> &v1, const R &v2 )
  {
    vector_td<typename vectorTDReturnType<T,R>::type,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = v1.vec[i]/v2;
    return res;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> operator/ ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  {
    vector_td<typename vectorTDReturnType<T,R>::type,D> res = v1;
    for(unsigned int i=0; i<D; i++ ) res[i] /= v2[i];
    return res;
  }

  template< class T, class R, unsigned int D > __inline__ 
  vector_td<typename vectorTDReturnType<T,R>::type,D> component_wise_div ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    return v1/v2;
  }

  // 
  // "Strong" comparison operators
  //

  template< class T, unsigned int D > __inline__ 
  bool operator== ( const vector_td<T,D> &v1, const vector_td<T,D> &v2 ) 
  { 
    for(unsigned int i=0; i<D; i++ ) if(!(v1.vec[i] == v2.vec[i])) return false;
    return true;
  }

  template< class T, unsigned int D > __inline__ 
  bool operator!= ( const vector_td<T,D> &v1, const vector_td<T,D> &v2 ) 
  { 
    for(unsigned int i=0; i<D; i++ ) if((v1.vec[i] != v2.vec[i])) return true;
    return false;
  }

  template< class T, unsigned int D > __inline__ 
  bool operator&& ( const vector_td<T,D> &v1, const vector_td<T,D> &v2 ) 
  { 
    for(unsigned int i=0; i<D; i++ ) if(!(v1.vec[i] && v2.vec[i])) return false;
    return true;
  }

  template< class T, unsigned int D > __inline__ 
  bool operator|| ( const vector_td<T,D> &v1, const vector_td<T,D> &v2 ) 
  { 
    for(unsigned int i=0; i<D; i++ ) if(!(v1.vec[i] || v2.vec[i])) return false;
    return true;
  }

  template< class T,class R, unsigned int D > __inline__ 
  bool operator< ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(!(v1.vec[i] < v2.vec[i])) return false;
    return true;
  }

  template< class T,class R, unsigned int D > __inline__ 
  bool operator<= ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(!(v1.vec[i] <= v2.vec[i])) return false;
    return true;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool operator> ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(!(v1.vec[i] > v2.vec[i])) return false;
    return true;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool operator>= ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(!(v1.vec[i] >= v2.vec[i])) return false;
    return true;
  }

  //
  // "Weak" comparison "operators"
  //

  template< class T, class R, unsigned int D > __inline__ 
  bool weak_equal ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(v1.vec[i] == v2.vec[i]) return true;
    return false;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool weak_not_equal ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(v1.vec[i] != v2.vec[i]) return true;
    return false;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool weak_and ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(v1.vec[i] && v2.vec[i]) return true;
    return false;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool weak_or ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(v1.vec[i] || v2.vec[i]) return true;
    return false;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool weak_less ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(v1.vec[i] < v2.vec[i]) return true;
    return false;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool weak_less_equal ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(v1.vec[i] <= v2.vec[i]) return true;
    return false;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool weak_greater ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(v1.vec[i] > v2.vec[i]) return true;
    return false;
  }

  template< class T, class R, unsigned int D > __inline__ 
  bool weak_greater_equal ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    for(unsigned int i=0; i<D; i++ ) if(v1.vec[i] >= v2.vec[i]) return true;
    return false;
  }

  //
  // Vector comparison "operators"
  //

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<bool,D> vector_equal ( const vector_td<T,D> &v1, const vector_td<T,D> &v2 )
  { 
    vector_td<T,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = (v1.vec[i] == v2.vec[i]);
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<bool,D> vector_not_equal ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    vector_td<T,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = (v1.vec[i] != v2.vec[i]);
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<T,D> vector_and ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
    vector_td<T,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = (v1.vec[i] && v2.vec[i]);
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<bool,D> vector_or ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
  	vector_td<bool,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = (v1.vec[i] || v2.vec[i]);
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<bool,D> vector_less ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
  	vector_td<bool,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = (v1.vec[i] < v2.vec[i]);
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<bool,D> vector_less_equal ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  { 
  	vector_td<bool,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = (v1.vec[i] <= v2.vec[i]);
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<bool,D> vector_greater ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  {
    vector_td<bool,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = (v1.vec[i] > v2.vec[i]);
    return res;
  }

  template< class T,class R, unsigned int D > __inline__ 
  vector_td<bool,D> vector_greater_equal ( const vector_td<T,D> &v1, const vector_td<R,D> &v2 )
  {  
  	vector_td<bool,D> res;
    for(unsigned int i=0; i<D; i++ ) res.vec[i] = (v1.vec[i] >= v2.vec[i]);
    return res;
  }

  //
  // Integer only operators
  //

  template< class T, unsigned int D > __inline__ 
  void operator<<= ( vector_td<T,D> &v1, size_t shifts ) 
  { 
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] <<= shifts;
  }

  template< class T, unsigned int D > __inline__ 
  void operator>>= ( vector_td<T,D> &v1, size_t shifts ) 
  { 
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] >>= shifts;
  }

  template< class T, unsigned int D > __inline__ 
  vector_td<T,D> operator<< ( const vector_td<T,D> &v1, size_t shifts ) 
  { 
    vector_td<T,D> res = v1;
    res <<= shifts;
    return res;
  }

  template< class T, unsigned int D > __inline__ 
  vector_td<T,D> operator>> ( const vector_td<T,D> &v1, size_t shifts ) 
  { 
    vector_td<T,D> res = v1;
    res >>= shifts;
    return res;
  }

  template< class T, unsigned int D > __inline__ 
  void operator%= ( vector_td<T,D> &v1, const vector_td<T,D> &v2 ) 
  { 
    for(unsigned int i=0; i<D; i++ ) v1.vec[i] %= v2.vec[i];
  }

  template< class T, unsigned int D > __inline__ 
  vector_td<T,D> operator% ( const vector_td<T,D> &v1, const vector_td<T,D> &v2 ) 
  { 
    vector_td<T,D> res = v1;
    res %= v2;
    return res;
  }


}

namespace std {

template<class T, unsigned int D> T* begin(Gadgetron::vector_td<T,D>& v){
  return v.vec;
};

template<class T, unsigned int D> T* end(Gadgetron::vector_td<T,D>& v){
return v.vec+D;
};

}

///////////////////////////////////////////////////////////////////////////////////////////////
///  NDArray
///////////////////////////////////////////////////////////////////////////////////////////////



/** \file NDArray.h
\brief Abstract base class for all Gadgetron host and device arrays
*/


// #include <new>
// #include <vector>
// #include <iostream>
// #include <stdexcept>
// #include <array>
// #include <algorithm>
// #include <cstdint>
// #include <numeric>

namespace Gadgetron{

    template <typename T> class NDArray
    {
    public:

        typedef T element_type;
        typedef T value_type;

        NDArray () : data_(0), elements_(0), delete_data_on_destruct_(true)
        {
        }

        virtual ~NDArray() {}

        virtual void create(const std::vector<size_t> &dimensions);
        virtual void create(const std::vector<size_t> &dimensions, T* data, bool delete_data_on_destruct = false);

        void squeeze();

        void reshape(const std::vector<size_t>& dims);

        /**
         * Reshapes the array to the given dimensions.
         * One of the dimensions can be -1, in which case that dimension will be calculated based on the other.
         * @param dims
         */
        void reshape(std::initializer_list<std::int64_t> dims);

        template<class... INDICES>
        std::enable_if_t<Core::all_of_v<std::is_integral_v<INDICES>...>> reshape(INDICES ... ind) {
            //static_assert(std::is_integral_v<INDICES> && ...);
            this->reshape(std::initializer_list<std::int64_t>{std::int64_t(ind)...});
        }

        bool dimensions_equal(const std::vector<size_t>* d) const { return this->dimensions_equal(*d); }
        bool dimensions_equal(const std::vector<size_t>& d) const;

        template<class S> bool dimensions_equal(const NDArray<S>* a) const { return this->dimensions_equal(*a); }
        template<class S> bool dimensions_equal(const NDArray<S>& a) const
        {
            std::vector<size_t> dim;
            a.get_dimensions(dim);

            if ( this->dimensions_.size() != dim.size() ) return false;

            size_t NDim = this->dimensions_.size();
            for ( size_t d=0; d<NDim; d++ )
            {
                if ( this->dimensions_[d] != dim[d] ) return false;
            }

            return true;
        }

        size_t get_number_of_dimensions() const;

        size_t get_size(size_t dimension) const;

        std::vector<size_t> get_dimensions() const;
        void get_dimensions(std::vector<size_t>& dim) const;

        std::vector<size_t> const &dimensions() const;

        const T* get_data_ptr() const;
        T* get_data_ptr();

        const T* data() const;
        T* data();

        size_t size() const;
        size_t get_number_of_elements() const;

        bool empty() const;

        size_t get_number_of_bytes() const;

        bool delete_data_on_destruct() const;
        void delete_data_on_destruct(bool d);

        size_t calculate_offset(const std::vector<size_t>& ind) const;
        static size_t calculate_offset(const std::vector<size_t>& ind, const std::vector<size_t>& offsetFactors);

        size_t calculate_offset(size_t x, size_t y) const;
        size_t calculate_offset(size_t x, size_t y, size_t z) const;
        size_t calculate_offset(size_t x, size_t y, size_t z, size_t s) const;
        size_t calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p) const;
        size_t calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r) const;
        size_t calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a) const;
        size_t calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q) const;
        size_t calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q, size_t u) const;

        size_t get_offset_factor(size_t dim) const;
        void get_offset_factor(std::vector<size_t>& offset) const;
        std::vector<size_t> get_offset_factor() const;

        size_t get_offset_factor_lastdim() const;

        void calculate_offset_factors(const std::vector<size_t>& dimensions);
        static void calculate_offset_factors(const std::vector<size_t>& dimensions, std::vector<size_t>& offsetFactors);

        std::vector<size_t> calculate_index( size_t offset ) const;
        void calculate_index( size_t offset, std::vector<size_t>& index ) const;
        static void calculate_index( size_t offset, const std::vector<size_t>& offsetFactors, std::vector<size_t>& index );

        void clear();

        /// whether a point is within the array range
        bool point_in_range(const std::vector<size_t>& ind) const;
        bool point_in_range(size_t x) const;
        bool point_in_range(size_t x, size_t y) const;
        bool point_in_range(size_t x, size_t y, size_t z) const;
        bool point_in_range(size_t x, size_t y, size_t z, size_t s) const;
        bool point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p) const;
        bool point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r) const;
        bool point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a) const;
        bool point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q) const;
        bool point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q, size_t u) const;

    protected:

        virtual void allocate_memory() = 0;
        virtual void deallocate_memory() = 0;

    protected:

        std::vector<size_t> dimensions_;
        std::array<size_t, 12> offsetFactors_;
        T* data_;
        size_t elements_;
        bool delete_data_on_destruct_;
    };

    template <typename T> 
    inline void NDArray<T>::create(const std::vector<size_t>& dimensions)
    {
        dimensions_ = dimensions;
        allocate_memory();
        calculate_offset_factors(dimensions_);
    }

    template <typename T> 
    void NDArray<T>::create(const std::vector<size_t> &dimensions, T* data, bool delete_data_on_destruct)
    {
        if (!data) throw std::runtime_error("NDArray<T>::create(): 0x0 pointer provided");    
        dimensions_ = dimensions;
        this->data_ = data;
        this->delete_data_on_destruct_ = delete_data_on_destruct;
        this->elements_ = 1;
        for (size_t i = 0; i < this->dimensions_.size(); i++){
            this->elements_ *= this->dimensions_[i];
        }
        calculate_offset_factors(dimensions_);
    }

    template <typename T> 
    inline void NDArray<T>::squeeze()
    {
        std::vector<size_t> new_dimensions;
        for (size_t i = 0; i < dimensions_.size(); i++){
            if (dimensions_[i] != 1){
                new_dimensions.push_back(dimensions_[i]);
            }
        }
        dimensions_ = new_dimensions;
        this->calculate_offset_factors(dimensions_);
    }

    template <typename T> 
    inline void NDArray<T>::reshape(const std::vector<size_t>& dims)
    {
        size_t new_elements = 1;
        for (size_t i = 0; i < dims.size(); i++){
            new_elements *= dims[i];
        }

        if (new_elements != elements_)
            throw std::runtime_error("NDArray<T>::reshape : Number of elements cannot change during reshape");    

        // Copy the input dimensions array
        dimensions_ = dims;
        this->calculate_offset_factors(dimensions_);
    }

    template <typename T> void NDArray<T>::reshape(std::initializer_list<std::int64_t> dims) {
        std::vector<std::int64_t> dim_vec(dims);
        auto negatives = std::count(dims.begin(), dims.end(), -1);
        if (negatives > 1)
            throw std::runtime_error("Only a single reshape dimension can be negative");

        if (negatives == 1) {
            auto elements    = std::accumulate(dims.begin(), dims.end(), -1, std::multiplies<std::int64_t>());
            auto neg_element = std::find(dim_vec.begin(), dim_vec.end(), -1);
            *neg_element     = this->elements_ / elements;
        }

        auto new_dims = std::vector<size_t>(dim_vec.begin(), dim_vec.end());
        this->reshape(new_dims);
    }

    template <typename T> 
    inline bool NDArray<T>::dimensions_equal(const std::vector<size_t>& d) const
    {
        if ( this->dimensions_.size() != d.size() ) return false;

        size_t NDim = this->dimensions_.size();
        for ( size_t ii=0; ii<NDim; ii++ )
        {
            if ( this->dimensions_[ii] != d[ii] ) return false;
        }

        return true;
    }

    template <typename T>
    inline size_t NDArray<T>::get_number_of_dimensions() const
    {
        return (size_t)dimensions_.size();
    }

    template <typename T> 
    inline size_t NDArray<T>::get_size(size_t dimension) const
    {
        if (dimension >= dimensions_.size()){
            return 1;
        }
        else{
            return dimensions_[dimension];
        }
    }

    template <typename T> 
    inline std::vector<size_t> NDArray<T>::get_dimensions() const
    {
        return this->dimensions_;
    }

    template <typename T> 
    inline void NDArray<T>::get_dimensions(std::vector<size_t>& dim) const
    {
        dim = dimensions_;
    }

    template<class T>
    inline std::vector<size_t> const &NDArray<T>::dimensions() const {
        return dimensions_;
    }


    template <typename T> 
    inline const T* NDArray<T>::get_data_ptr() const
    { 
        return data_;
    }
    template <typename T>
    inline T* NDArray<T>::get_data_ptr()
    {
        return data_;
    }

    template <typename T>
    const T* NDArray<T>::data() const
    {
        return data_;
    }

    template <typename T>
    T* NDArray<T>::data()
    {
        return data_;
    }


    template<class T>
    inline size_t NDArray<T>::size() const
    {
        return elements_;
    }

    template<class T>
    inline bool NDArray<T>::empty() const
    {
        return elements_ == 0;
    }

    template <class T>
    inline size_t NDArray<T>::get_number_of_elements() const {
        return size();
    }

    template <typename T> 
    inline size_t NDArray<T>::get_number_of_bytes() const
    {
        return elements_*sizeof(T);
    }

    template <typename T> 
    inline bool NDArray<T>::delete_data_on_destruct() const
    {
        return delete_data_on_destruct_;
    }

    template <typename T> 
    inline void NDArray<T>::delete_data_on_destruct(bool d)
    {
        delete_data_on_destruct_ = d;
    }

    template <typename T> 
    size_t NDArray<T>::calculate_offset(const std::vector<size_t>& ind, const std::vector<size_t>& offsetFactors)
    {
        size_t offset = ind[0];

        for( size_t i = 1; i < ind.size(); i++ )
        {
            offset += ind[i] * offsetFactors[i];
        }

        return offset;
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(const std::vector<size_t>& ind) const
    {
        size_t offset = ind[0];
        for( size_t i = 1; i < dimensions_.size(); i++ )
            offset += ind[i] * offsetFactors_[i];
        return offset;
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(size_t x, size_t y) const
    {
//        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==2);
        return x + y * offsetFactors_[1];
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(size_t x, size_t y, size_t z) const
    {
//        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==3);
        return x + y * offsetFactors_[1] + z * offsetFactors_[2];
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(size_t x, size_t y, size_t z, size_t s) const
    {
//        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==4);
        return x + y * offsetFactors_[1] + z * offsetFactors_[2] + s * offsetFactors_[3];
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p) const
    {
//        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==5);
        return x + y * offsetFactors_[1] + z * offsetFactors_[2] + s * offsetFactors_[3] + p * offsetFactors_[4];
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r) const
    {
//        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==6);
        return x + y * offsetFactors_[1] + z * offsetFactors_[2] + s * offsetFactors_[3] + p * offsetFactors_[4] + r * offsetFactors_[5];
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a) const
    {
//        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==7);
        return x + y * offsetFactors_[1] + z * offsetFactors_[2] + s * offsetFactors_[3] + p * offsetFactors_[4] + r * offsetFactors_[5] + a * offsetFactors_[6];
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q) const
    {
//        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==8);
        return x + y * offsetFactors_[1] + z * offsetFactors_[2] + s * offsetFactors_[3] + p * offsetFactors_[4] + r * offsetFactors_[5] + a * offsetFactors_[6] + q * offsetFactors_[7];
    }

    template <typename T> 
    inline size_t NDArray<T>::calculate_offset(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q, size_t u) const
    {
//        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==9);
        return x + y * offsetFactors_[1] + z * offsetFactors_[2] + s * offsetFactors_[3] + p * offsetFactors_[4] + r * offsetFactors_[5] + a * offsetFactors_[6] + q * offsetFactors_[7]+ u * offsetFactors_[8];
    }

    template <typename T> 
    inline size_t NDArray<T>::get_offset_factor(size_t dim) const
    {
        if ( dim >= dimensions_.size() )
            throw std::runtime_error("NDArray<T>::get_offset_factor : index out of range");
        return offsetFactors_[dim];
    }

    template <typename T> 
    inline void NDArray<T>::get_offset_factor(std::vector<size_t>& offset) const
    {
        offset=std::vector<size_t>(offsetFactors_.begin(),offsetFactors_.end());
    }

    template <typename T> 
    inline size_t NDArray<T>::get_offset_factor_lastdim() const
    {
        if( dimensions_.size() == 0 )
            throw std::runtime_error("NDArray<T>::get_offset_factor_lastdim : array is empty");

        return get_offset_factor(dimensions_.size()-1);
    }

    template <typename T> 
    inline std::vector<size_t> NDArray<T>::get_offset_factor() const
    {
        size_t NDim = this->dimensions_.size();
        std::vector<size_t> res(NDim, 0);
        for (auto i = 0; i < NDim; i++) res[i] = this->offsetFactors_[i];
        return res;
    }

    template <typename T> 
    void NDArray<T>::calculate_offset_factors(const std::vector<size_t>& dimensions, std::vector<size_t>& offsetFactors)
    {
        offsetFactors.resize(dimensions.size());
        for( size_t i = 0; i < dimensions.size(); i++ )
        {
            size_t k = 1;
            for( size_t j = 0; j < i; j++ )
            {
                k *= dimensions[j];
            }

            offsetFactors[i] = k;
        }
    }

    template <typename T> 
    inline void NDArray<T>::calculate_offset_factors(const std::vector<size_t>& dimensions)
    {
        std::fill(offsetFactors_.begin(),offsetFactors_.end(),1);
        size_t a = dimensions.size();
        size_t b = offsetFactors_.size();
        size_t offsets = a<b ? a : b;
        for( size_t i = 0; i < offsets; i++ ){
            size_t k = 1;
            for( size_t j = 0; j < i; j++ )
                k *= dimensions[j];
            offsetFactors_[i] = k;
        }
    }

    template <typename T> 
    inline std::vector<size_t> NDArray<T>::calculate_index( size_t offset ) const
    {
        if( dimensions_.size() == 0 )
            throw std::runtime_error("NDArray<T>::calculate_index : array is empty");

        std::vector<size_t> index(dimensions_.size());
        for( long long i = dimensions_.size()-1; i>=0; i-- ){
            index[i] = offset / offsetFactors_[i];
            offset %= offsetFactors_[i];
        }
        return index;
    }

    template <typename T> 
    inline void NDArray<T>::calculate_index( size_t offset, std::vector<size_t>& index ) const
    {
        if( dimensions_.size() == 0 )
            throw std::runtime_error("NDArray<T>::calculate_index : array is empty");

        index.resize(dimensions_.size(), 0);
        for( long long i = dimensions_.size()-1; i>=0; i-- ){
            index[i] = offset / offsetFactors_[i];
            offset %= offsetFactors_[i];
        }
    }

    template <typename T> 
    void NDArray<T>::calculate_index( size_t offset, const std::vector<size_t>& offsetFactors, std::vector<size_t>& index )
    {
        index.resize(offsetFactors.size(), 0);

        for( long long i = offsetFactors.size()-1; i>=0; i-- )
        {
            index[i] = offset / offsetFactors[i];
            offset %= offsetFactors[i];
        }
    }

    template <typename T> 
    void NDArray<T>::clear()
    {
        if ( this->delete_data_on_destruct_ ){
            this->deallocate_memory();
        } else{
            throw std::runtime_error("NDArray<T>::clear : trying to reallocate memory not owned by array.");
        }

        this->data_ = 0;
        this->elements_ = 0;
        this->dimensions_.clear();
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(const std::vector<size_t>& ind) const
    {
        unsigned int D = dimensions_.size();
        if ( ind.size() != D ) return false;

        unsigned int ii;
        for ( ii=0; ii<D; ii++ )
        {
            if ( ind[ii]>=dimensions_[ii] )
            {
                return false;
            }
        }

        return true;
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==1);
        return (x<dimensions_[0]);
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x, size_t y) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==2);
        return ((x<dimensions_[0]) && (y<dimensions_[1]));
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x, size_t y, size_t z) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==3);
        return ( (x<dimensions_[0]) && (y<dimensions_[1]) && (z<dimensions_[2]));
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x, size_t y, size_t z, size_t s) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==4);
        return ( (x<dimensions_[0]) && (y<dimensions_[1]) && (z<dimensions_[2]) && (s<dimensions_[3]));
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==5);
        return ( (x<dimensions_[0]) && (y<dimensions_[1]) && (z<dimensions_[2]) && (s<dimensions_[3]) && (p<dimensions_[4]));
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==6);
        return ( (x<dimensions_[0]) && (y<dimensions_[1]) && (z<dimensions_[2]) && (s<dimensions_[3]) && (p<dimensions_[4]) && (r<dimensions_[5]));
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==7);
        return ( (x<dimensions_[0]) && (y<dimensions_[1]) && (z<dimensions_[2]) && (s<dimensions_[3]) && (p<dimensions_[4]) && (r<dimensions_[5]) && (a<dimensions_[6]));
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==8);
        return ( (x<dimensions_[0]) && (y<dimensions_[1]) && (z<dimensions_[2]) && (s<dimensions_[3]) && (p<dimensions_[4]) && (r<dimensions_[5]) && (a<dimensions_[6]) && (q<dimensions_[7]));
    }

    template <typename T> 
    inline bool NDArray<T>::point_in_range(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q, size_t u) const
    {
        GADGET_DEBUG_CHECK_THROW(dimensions_.size()==9);
        return ( (x<dimensions_[0]) && (y<dimensions_[1]) && (z<dimensions_[2]) && (s<dimensions_[3]) && (p<dimensions_[4]) && (r<dimensions_[5]) && (a<dimensions_[6]) && (q<dimensions_[7]) && (u<dimensions_[8]));
    }




}






///////////////////////////////////////////////////////////////////////////////////////////////
///  hoNDArray
///////////////////////////////////////////////////////////////////////////////////////////////

/** \file hoNDArray.h
    \brief CPU-based N-dimensional array (data container)
*/

namespace Gadgetron{

    namespace Indexing {
        class Slice {};
        constexpr auto slice = Slice{};
    }
   template<class... ARGS>
   struct ValidIndex : std::integral_constant<bool, Core::all_of_v<Core::is_convertible_v<ARGS,size_t>...>> {};

   template<> struct ValidIndex<> : std::true_type {};

   template<class... ARGS>
   struct ValidIndex<Indexing::Slice,ARGS...> : ValidIndex<ARGS...> {};


    namespace {
       namespace gadgetron_detail {

           template <size_t count, class... ARGS> struct count_slices { static constexpr size_t value = count; };

           template <size_t count, class... ARGS>
           struct count_slices<count, Indexing::Slice, ARGS...> : count_slices<count + 1, ARGS...> {};

           template <size_t count, class T, class... ARGS>
           struct count_slices<count, T, ARGS...> : count_slices<count, ARGS...> {};

           template <class... ARGS> struct is_contiguous_index {
                static constexpr bool value = true;
           };
           template<class T, class... ARGS> struct is_contiguous_index<T,ARGS...>{

               static constexpr bool value = !Core::any_of_v<Core::is_same_v<Indexing::Slice,ARGS>...>;
           };

           template<class... ARGS> struct is_contiguous_index<Indexing::Slice,ARGS...> {
               static constexpr bool value = is_contiguous_index<ARGS...>::value;
           };

           static_assert(is_contiguous_index<Indexing::Slice>::value);
           static_assert(is_contiguous_index<Indexing::Slice,size_t>::value);
           static_assert(is_contiguous_index<Indexing::Slice,long long >::value);
           static_assert(is_contiguous_index<Indexing::Slice,Indexing::Slice,size_t>::value);
           static_assert(is_contiguous_index<Indexing::Slice,Indexing::Slice,long long>::value);
           static_assert(!is_contiguous_index<size_t ,Indexing::Slice,Indexing::Slice,size_t>::value);
           static_assert(!is_contiguous_index<int,Indexing::Slice,Indexing::Slice,size_t>::value);
           static_assert(!is_contiguous_index<long long,Indexing::Slice,Indexing::Slice,size_t>::value);

       }
    }
   template<class T> class hoNDArray;


   template<class T, size_t D, bool contigous = false>
   class hoNDArrayView {
   public:
       hoNDArrayView& operator=(const hoNDArrayView<T,D,!contigous>&);
       hoNDArrayView& operator=(const hoNDArrayView&);
       hoNDArrayView<T,D,contigous>& operator=(const hoNDArray<T>&);

        template<class... INDICES>
       std::enable_if_t<Core::all_of_v<Core::is_convertible_v<INDICES,size_t>...> && (sizeof...(INDICES) == D),T&>
       operator()(INDICES... indices);

       template<class... INDICES>
       std::enable_if_t<Core::all_of_v<Core::is_convertible_v<INDICES,size_t>...> && (sizeof...(INDICES) == D),const T&>
       operator()(INDICES... indices) const;


       template<typename Dummy1 = void, typename  = std::enable_if_t<contigous,Dummy1>>
       operator hoNDArray<T>();
       operator const hoNDArray<T>() const;



   private:
       friend class hoNDArray<T>;
       friend class hoNDArrayView<T,D,!contigous>;
       hoNDArrayView(const std::array<size_t,D>& strides, const std::array<size_t,D>& dimensions, T*);

       vector_td<size_t, D> strides;
       vector_td<size_t, D> dimensions;
       T* data;

   };



  template <typename T> class hoNDArray : public NDArray<T>
  {
  public:

    typedef NDArray<T> BaseClass;
    typedef float coord_type;
    typedef T value_type;
    using iterator = T*;
    using const_iterator = const T*;

    hoNDArray();

    explicit hoNDArray(const std::vector<size_t>& dimensions);
    hoNDArray(const std::vector<size_t>& dimensions, T* data, bool delete_data_on_destruct = false);

    hoNDArray(std::initializer_list<size_t> dimensions);
    hoNDArray(std::initializer_list<size_t> dimensions,T* data, bool delete_data_on_destruct = false);

    explicit hoNDArray(size_t len);
    hoNDArray(size_t sx, size_t sy);
    hoNDArray(size_t sx, size_t sy, size_t sz);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss);

    hoNDArray(size_t len, T* data, bool delete_data_on_destruct = false);
    hoNDArray(size_t sx, size_t sy, T* data, bool delete_data_on_destruct = false);
    hoNDArray(size_t sx, size_t sy, size_t sz, T* data, bool delete_data_on_destruct = false);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, T* data, bool delete_data_on_destruct = false);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, T* data, bool delete_data_on_destruct = false);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, T* data, bool delete_data_on_destruct = false);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, T* data, bool delete_data_on_destruct = false);
    hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss, T* data, bool delete_data_on_destruct = false);

    virtual ~hoNDArray();

    // Copy constructors
    hoNDArray(const hoNDArray<T> &a);

    template<class S>
    explicit hoNDArray(const hoNDArray<S>& other);

    //Move constructors
    hoNDArray(hoNDArray<T>&& a) noexcept;
    hoNDArray& operator=(hoNDArray&& rhs) noexcept;

    // Assignment operator
    hoNDArray& operator=(const hoNDArray& rhs);

    template<unsigned int D, bool C>
    hoNDArray& operator=(const hoNDArrayView<T,D,C>& view);

    bool operator==(const hoNDArray& rhs) const;

    virtual void create(const std::vector<size_t>& dimensions);
    virtual void create(const std::vector<size_t> &dimensions, T* data, bool delete_data_on_destruct = false);

    virtual void create(std::initializer_list<size_t> dimensions);
    virtual void create(std::initializer_list<size_t> dimensions,T* data, bool delete_data_on_destruct = false);

    virtual void create(size_t len);
    virtual void create(size_t sx, size_t sy);
    virtual void create(size_t sx, size_t sy, size_t sz);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss, size_t su);

    virtual void create(size_t len, T* data, bool delete_data_on_destruct = false);
    virtual void create(size_t sx, size_t sy, T* data, bool delete_data_on_destruct = false);
    virtual void create(size_t sx, size_t sy, size_t sz, T* data, bool delete_data_on_destruct = false);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, T* data, bool delete_data_on_destruct = false);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, T* data, bool delete_data_on_destruct = false);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, T* data, bool delete_data_on_destruct = false);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, T* data, bool delete_data_on_destruct = false);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss, T* data, bool delete_data_on_destruct = false);
    virtual void create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss, size_t su, T* data, bool delete_data_on_destruct = false);

    T& operator()( const std::vector<size_t>& ind );
    const T& operator()( const std::vector<size_t>& ind ) const;

    T& operator()( size_t x );
    const T& operator()( size_t x ) const;

    T& operator()( size_t x, size_t y );
    const T& operator()( size_t x, size_t y ) const;

    T& operator()( size_t x, size_t y, size_t z );
    const T& operator()( size_t x, size_t y, size_t z ) const;

    T& operator()( size_t x, size_t y, size_t z, size_t s );
    const T& operator()( size_t x, size_t y, size_t z, size_t s ) const;

    T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p );
    const T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p ) const;

    T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p, size_t r );
    const T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p, size_t r ) const;

    T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a );
    const T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a ) const;

    T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q );
    const T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q ) const;

    T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q, size_t u );
    const T& operator()( size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q, size_t u ) const;

    template<class... INDICES, class = std::enable_if_t<Core::any_of_v<Core::is_same_v<INDICES,Indexing::Slice>...>> >
    auto operator()(const INDICES&... );

    template<class... INDICES, class = std::enable_if_t<Core::any_of_v<Core::is_same_v<INDICES,Indexing::Slice>...>> >
    auto operator()(const INDICES&... ) const -> const hoNDArrayView<T,gadgetron_detail::count_slices<0, INDICES...>::value, gadgetron_detail::is_contiguous_index<INDICES...>::value>;

    void fill(T value);

    T* begin();
    const T* begin() const;

    T* end();
    const T* end() const;

    T& at( size_t idx );
    const T& at( size_t idx ) const;

    T& operator[]( size_t idx );
    const T& operator[]( size_t idx ) const;
    //T& operator()( size_t idx );
    //const T& operator()( size_t idx ) const;

    //T& operator()( const std::vector<size_t>& ind );
    //const T& operator()( const std::vector<size_t>& ind ) const;

    template<typename T2> 
    bool copyFrom(const hoNDArray<T2>& aArray)
    {
        try
        {
            if (!this->dimensions_equal(aArray))
            {
                this->create(aArray.dimensions());
            }

            long long i;
#pragma omp parallel for default(none) private(i) shared(aArray)
            for (i = 0; i < (long long)elements_; i++)
            {
                data_[i] = static_cast<T>(aArray(i));
            }
        }
        catch (...)
        {
            GERROR_STREAM("Exceptions happened in hoNDArray::copyFrom(...) ... ");
            return false;
        }
        return true;
    }

    void get_sub_array(const std::vector<size_t>& start, std::vector<size_t>& size, hoNDArray<T>& out) const;

    virtual void print(std::ostream& os) const;
    virtual void printContent(std::ostream& os) const;

    [[deprecated("Use IO::write instead")]]
    virtual bool serialize(char*& buf, size_t& len) const;
    [[deprecated("Use IO::read instead")]]
    virtual bool deserialize(char* buf, size_t& len);

  protected:

    using BaseClass::dimensions_;
    using BaseClass::offsetFactors_;
    using BaseClass::data_;
    using BaseClass::elements_;
    using BaseClass::delete_data_on_destruct_;

    virtual void allocate_memory();
    virtual void deallocate_memory();

    // Generic allocator / deallocator
    //

    template<class X> void _allocate_memory( size_t size, X** data )
    {
      *data = new X[size];
    }

    template<class X> void _deallocate_memory( X* data )
    {
      delete [] data;
    }


  };

}


namespace Gadgetron {
    template<typename T>
    hoNDArray<T>::hoNDArray() : Gadgetron::NDArray<T>::NDArray() {}

    template<typename T>
    hoNDArray<T>::hoNDArray(const std::vector<size_t> &dimensions) : Gadgetron::NDArray<T>::NDArray() {
        this->create(dimensions);
    }

    template<class T>
    hoNDArray<T>::hoNDArray(std::initializer_list<size_t> dimensions) {
        this->create(dimensions);
    }

    template<class T>
    hoNDArray<T>::hoNDArray(std::initializer_list<size_t> dimensions, T *data, bool delete_data_on_destruct) {
        this->create(dimensions, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t len) : Gadgetron::NDArray<T>::NDArray() {
        std::vector<size_t> dim(1);
        dim[0] = len;
        this->create(dim);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy) : Gadgetron::NDArray<T>::NDArray() {
        std::vector<size_t> dim(2);
        dim[0] = sx;
        dim[1] = sy;
        this->create(dim);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz) : Gadgetron::NDArray<T>::NDArray() {
        std::vector<size_t> dim(3);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        this->create(dim);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, size_t st) : Gadgetron::NDArray<T>::NDArray() {
        std::vector<size_t> dim(4);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        this->create(dim);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp) : Gadgetron::NDArray<T>::NDArray() {
        std::vector<size_t> dim(5);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        this->create(dim);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq) : Gadgetron::NDArray<T>::NDArray() {
        std::vector<size_t> dim(6);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        this->create(dim);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr)
            : Gadgetron::NDArray<T>::NDArray() {
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

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss)
            : Gadgetron::NDArray<T>::NDArray() {
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

    template<typename T>
    hoNDArray<T>::hoNDArray(const std::vector<size_t> &dimensions, T *data, bool delete_data_on_destruct)
            : Gadgetron::NDArray<T>::NDArray() {
        this->create(dimensions, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t len, T *data, bool delete_data_on_destruct) : Gadgetron::NDArray<T>::NDArray() {
        this->create(len, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, T *data, bool delete_data_on_destruct) : Gadgetron::NDArray<T>::NDArray() {
        this->create(sx,sy, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, T *data, bool delete_data_on_destruct)
            : Gadgetron::NDArray<T>::NDArray() {
   ;
        this->create(sx,sy,sz, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, T *data, bool delete_data_on_destruct)
            : Gadgetron::NDArray<T>::NDArray() {
        this->create(sx,sy,sz,st, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(
            size_t sx, size_t sy, size_t sz, size_t st, size_t sp, T *data, bool delete_data_on_destruct)
            : Gadgetron::NDArray<T>::NDArray() {
        this->create(sx,sy,sz,st,sp, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(
            size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, T *data, bool delete_data_on_destruct)
            : Gadgetron::NDArray<T>::NDArray() {
        this->create(sx,sy,sz,st,sp,sq, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, T *data,
                            bool delete_data_on_destruct)
            : Gadgetron::NDArray<T>::NDArray() {
        this->create(sx,sy,sz,st,sp,sq,sr, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss,
                            T *data, bool delete_data_on_destruct)
            : Gadgetron::NDArray<T>::NDArray() {
        std::vector<size_t> dim(8);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        dim[7] = ss;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    hoNDArray<T>::~hoNDArray() {
        if (this->delete_data_on_destruct_) {
            deallocate_memory();
        }
    }

    template<typename T>
    hoNDArray<T>::hoNDArray(const hoNDArray<T> &a) {
        this->data_ = 0;
        this->dimensions_ = a.dimensions_;
        offsetFactors_ = a.offsetFactors_;

        if (!this->dimensions_.empty()) {
            this->allocate_memory();
            std::copy(a.begin(),a.end(),this->begin());
        } else {
            this->elements_ = 0;
        }
    }

    template<typename T>
    template<class S>
    hoNDArray<T>::hoNDArray(const hoNDArray<S> &a) {
        this->create(a.dimensions());

        if (!this->dimensions_.empty()) {
            std::copy(a.begin(),a.end(),this->begin());
        }

    }


    template<typename T>
    hoNDArray<T>::hoNDArray(hoNDArray<T> &&a) noexcept : Gadgetron::NDArray<T>::NDArray() {
        data_ = a.data_;
        this->dimensions_ = a.dimensions_;
        this->elements_ = a.elements_;
        a.data_ = nullptr;
        this->offsetFactors_ = a.offsetFactors_;
        this->delete_data_on_destruct_ = a.delete_data_on_destruct_;
    }



    template<typename T>
    hoNDArray<T> &hoNDArray<T>::operator=(const hoNDArray<T> &rhs) {
        if (&rhs == this)
            return *this;

        if (rhs.get_number_of_elements() == 0) {
            this->clear();
            return *this;
        }

        // Are the dimensions the same? Then we can just memcpy
        if (!this->dimensions_equal(rhs)) {
            deallocate_memory();
            this->data_ = 0;
            this->dimensions_ = rhs.dimensions_;
            offsetFactors_ = rhs.offsetFactors_;
            allocate_memory();
        }
        std::copy(rhs.begin(),rhs.end(),this->begin());
        return *this;
    }



    template<typename T>
    hoNDArray<T> &hoNDArray<T>::operator=(hoNDArray<T> &&rhs) noexcept {
        if (&rhs == this)
            return *this;
        if (!this->delete_data_on_destruct_){ //We want to use the copy constructor if we don't own the data.
            hoNDArray<T>& rhs_ref = rhs;
            *this = rhs_ref;
            return *this;
        }
        this->clear();
        this->dimensions_ = rhs.dimensions_;
        this->offsetFactors_ = rhs.offsetFactors_;
        this->elements_ = rhs.elements_;
        data_ = rhs.data_;
        rhs.data_ = nullptr;
        this->delete_data_on_destruct_ = rhs.delete_data_on_destruct_;
        return *this;
    }



    template<typename T>
    void hoNDArray<T>::create(const std::vector<size_t> &dimensions) {
        if (this->dimensions_equal(dimensions)) {
            return;
        }

        this->clear();
        BaseClass::create(dimensions);
    }

    template<typename T>
    void hoNDArray<T>::create(const std::vector<size_t> &dimensions, T *data, bool delete_data_on_destruct) {
        if (!data)
            throw std::runtime_error("hoNDArray<T>::create(): 0x0 pointer provided");

        if (this->dimensions_equal(dimensions)) {
            if (this->delete_data_on_destruct_) {
                this->deallocate_memory();
            }

            this->data_ = data;
            this->delete_data_on_destruct_ = delete_data_on_destruct;
        } else {
            if (this->delete_data_on_destruct_) {
                this->deallocate_memory();
                this->data_ = NULL;
            }

            BaseClass::create(dimensions, data, delete_data_on_destruct);
        }
    }

    template<class T>
    void hoNDArray<T>::create(std::initializer_list<size_t> dimensions) {
        std::vector<size_t> dims(dimensions);
        this->create(dims);
    }

    template<class T>
    void hoNDArray<T>::create(std::initializer_list<size_t> dimensions, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dims(dimensions);
        this->create(dims, data, delete_data_on_destruct);
    }


    template<typename T>
    inline void hoNDArray<T>::create(size_t len) {
        std::vector<size_t> dim(1);
        dim[0] = len;
        this->create(dim);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy) {
        std::vector<size_t> dim(2);
        dim[0] = sx;
        dim[1] = sy;
        this->create(dim);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz) {
        std::vector<size_t> dim(3);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        this->create(dim);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st) {
        std::vector<size_t> dim(4);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        this->create(dim);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp) {
        std::vector<size_t> dim(5);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        this->create(dim);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq) {
        std::vector<size_t> dim(6);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        this->create(dim);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr) {
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

    template<typename T>
    inline void hoNDArray<T>::create(
            size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss) {
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

    template<typename T>
    inline void hoNDArray<T>::create(
            size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr, size_t ss, size_t su) {
        std::vector<size_t> dim(9);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        dim[7] = ss;
        dim[8] = su;
        this->create(dim);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t len, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(1);
        dim[0] = len;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(2);
        dim[0] = sx;
        dim[1] = sy;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(3);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    inline void hoNDArray<T>::create(
            size_t sx, size_t sy, size_t sz, size_t st, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(4);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    inline void hoNDArray<T>::create(
            size_t sx, size_t sy, size_t sz, size_t st, size_t sp, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(5);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    inline void hoNDArray<T>::create(
            size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(6);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr,
                                     T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(7);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr,
                                     size_t ss, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(8);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        dim[7] = ss;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    inline void hoNDArray<T>::create(size_t sx, size_t sy, size_t sz, size_t st, size_t sp, size_t sq, size_t sr,
                                     size_t ss, size_t su, T *data, bool delete_data_on_destruct) {
        std::vector<size_t> dim(9);
        dim[0] = sx;
        dim[1] = sy;
        dim[2] = sz;
        dim[3] = st;
        dim[4] = sp;
        dim[5] = sq;
        dim[6] = sr;
        dim[7] = ss;
        dim[8] = su;
        this->create(dim, data, delete_data_on_destruct);
    }

    template<typename T>
    void hoNDArray<T>::fill(T value) {
        std::fill(this->get_data_ptr(), this->get_data_ptr() + this->get_number_of_elements(), value);
    }

    template<typename T>
    inline T *hoNDArray<T>::begin() {
        return this->data_;
    }

    template<typename T>
    inline const T *hoNDArray<T>::begin() const {
        return this->data_;
    }

    template<typename T>
    inline T *hoNDArray<T>::end() {
        return (this->data_ + this->elements_);
    }

    template<typename T>
    inline const T *hoNDArray<T>::end() const {
        return (this->data_ + this->elements_);
    }

    template<typename T>
    inline T &hoNDArray<T>::at(size_t idx) {
        /*if( idx >= this->get_number_of_elements() )
        {
        BOOST_THROW_EXCEPTION( runtime_error("hoNDArray::at(): index out of range."));
        }*/
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->get_data_ptr()[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::at(size_t idx) const {
        /*if( idx >= this->get_number_of_elements() )
        {
        BOOST_THROW_EXCEPTION( runtime_error("hoNDArray::at(): index out of range."));
        }*/
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->get_data_ptr()[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator[](size_t idx) {
        /*if( idx >= this->get_number_of_elements() )
        {
        BOOST_THROW_EXCEPTION( runtime_error("hoNDArray::operator[]: index out of range."));
        }*/
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->get_data_ptr()[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator[](size_t idx) const {
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->get_data_ptr()[idx];
    }

    template<typename T>
    void hoNDArray<T>::get_sub_array(
            const std::vector<size_t> &start, std::vector<size_t> &size, hoNDArray<T> &out) const {
        if (start.size() != size.size()) {
            BOOST_THROW_EXCEPTION(runtime_error("hoNDArray<>::get_sub_array failed"));
        }

        if (start.size() != dimensions_.size()) {
            BOOST_THROW_EXCEPTION(runtime_error("hoNDArray<>::get_sub_array failed"));
        }

        out.create(&size);

        if (out.get_number_of_elements() == this->get_number_of_elements()) {
            out = *this;
            return;
        }

        std::vector<size_t> end(start.size());

        size_t ii;
        for (ii = 0; ii < start.size(); ii++) {
            end[ii] = start[ii] + size[ii] - 1;
            if (end[ii] >= dimensions_[ii]) {
                BOOST_THROW_EXCEPTION(runtime_error("hoNDArray<>::get_sub_array failed"));
            }
        }

        // Since doing memcpy for a chunk of contiguous data is much faster than
        //   copying element by element, we will loop with a single index
        //   through all the dimensions except the first one and memcpy the
        //   corresponding lines of data:

        size_t dim2D = out.get_number_of_elements() / size[0];

        // loop through all the dimensions except the first one:
        for (ii = 0; ii < dim2D; ii++) {
            // (linear) index corresponding to (0, ii) if "out" were 2D
            size_t ind1D = ii * size[0];

            // the indices i,j,k,... corresponding to that ind1D
            //    (note that the first index should be 0!):
            std::vector<size_t> ind = out.calculate_index(ind1D);

            // calculate the indices in the source array corresponding
            //   to "ind" in the output array:
            for (size_t jj = 0; jj < start.size(); jj++) {
                ind[jj] += start[jj];
            }
            // now, copy size[0] elements:
            std::copy_n(&((*this)(ind)), size[0], &out.data_[ind1D]);
        }
    }

    template<typename T>
    void hoNDArray<T>::printContent(std::ostream &os) const {
        using namespace std;

        os.unsetf(std::ios::scientific);
        os.setf(ios::fixed);

        size_t i;

        os << "Array dimension is : " << dimensions_.size() << endl;

        os << "Array size is : ";
        for (i = 0; i < dimensions_.size(); i++)
            os << dimensions_[i] << " ";
        os << endl;

        int elemTypeSize = sizeof(T);
        std::string elemTypeName = std::string(typeid(T).name());

        os << "Array data type is : " << elemTypeName << std::endl;
        os << "Byte number for each element is : " << elemTypeSize << std::endl;
        os << "Number of array size in bytes is : ";
        os << elements_ * elemTypeSize << std::endl;
        os << "Delete data on destruction flag is : " << this->delete_data_on_destruct_ << endl;

        // os << "-------------------------------------------" << std::endl;
        // size_t numOfPrints = 20;
        // if ( this->elements_ < numOfPrints ) numOfPrints = this->elements_;
        // for (i=0; i<numOfPrints; i++)
        //{
        //    os << i << " = " << (*this)(i) << std::endl;
        //}
        // os << "-------------------------------------------" << std::endl;

        os << std::endl;
    }

    template<typename T>
    void hoNDArray<T>::print(std::ostream &os) const {
        using namespace std;

        os.unsetf(std::ios::scientific);
        os.setf(ios::fixed);

        os << "--------------Gagdgetron ND Array -------------" << endl;
        this->printContent(os);
    }

    template<typename T>
    void hoNDArray<T>::allocate_memory() {
        deallocate_memory();

        if (!this->dimensions_.empty()) {
            this->elements_ = this->dimensions_[0];
            for (size_t i = 1; i < this->dimensions_.size(); i++) {
                this->elements_ *= this->dimensions_[i];
            }

            if (this->elements_ > 0) {
                this->_allocate_memory(this->elements_, &this->data_);

                if (this->data_ == 0x0) {
                    BOOST_THROW_EXCEPTION(bad_alloc());
                }

                this->delete_data_on_destruct_ = true;

                // memset(this->data_, 0, sizeof(T)*this->elements_);
            }
        } else {
            this->elements_ = 0;
        }
    }

    template<typename T>
    void hoNDArray<T>::deallocate_memory() {
        if (!(this->delete_data_on_destruct_)) {
            return;
        }

        if (this->data_) {
            this->_deallocate_memory(this->data_);
            this->data_ = 0x0;
        }
    }


    template<typename T>

    bool hoNDArray<T>::serialize(char *&buf, size_t &len) const {
        if (!Gadgetron::Core::is_trivially_copyable_v<T> ) throw std::runtime_error("Serialize only works for trivial types");

        if (buf != NULL)
            delete[] buf;

        size_t NDim = dimensions_.size();

        // number of dimensions + dimension vector + contents
        len = sizeof(size_t) + sizeof(size_t) * NDim + sizeof(T) * elements_;

        buf = new char[len];


        memcpy(buf, &NDim, sizeof(size_t));
        if (NDim > 0) {
            memcpy(buf + sizeof(size_t), &(dimensions_[0]), sizeof(size_t) * NDim);
            memcpy(buf + sizeof(size_t) + sizeof(size_t) * NDim, this->data_, sizeof(T) * elements_);
        }

        return true; // Temporary. Should not be a boolean function.
    }

    template<typename T>
    bool hoNDArray<T>::deserialize(char *buf, size_t &len) {
        if (!Gadgetron::Core::is_trivially_copyable_v<T> ) throw std::runtime_error("deserialize only works for trivial types");
        size_t NDim;
        memcpy(&NDim, buf, sizeof(size_t));

        if (NDim > 0) {
            std::vector<size_t> dimensions(NDim);
            memcpy(&dimensions[0], buf + sizeof(size_t), sizeof(size_t) * NDim);

            // allocate memory
            this->create(dimensions);

            // copy the content
            memcpy(this->data_, buf + sizeof(size_t) + sizeof(size_t) * NDim, sizeof(T) * elements_);
        } else {
            this->clear();
        }

        len = sizeof(size_t) + sizeof(size_t) * NDim + sizeof(T) * elements_;
        return true; // Temporary. Should not be a boolean function.
    }

    template<typename T>
    bool hoNDArray<T>::operator==(const hoNDArray &rhs) const {
        auto result = this->dimensions_equal(rhs.dimensions());
        if (!result)
            return false;
        for (size_t i = 0; i < this->size(); i++)
            result &= this->data_[i] == rhs[i];
        return result;
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(const std::vector<size_t> &ind) {
        size_t idx = this->calculate_offset(ind);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(const std::vector<size_t> &ind) const {
        size_t idx = this->calculate_offset(ind);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(size_t x) {
        GADGET_DEBUG_CHECK_THROW(x < this->get_number_of_elements());
        return this->data_[x];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(size_t x) const {
        GADGET_DEBUG_CHECK_THROW(x < this->get_number_of_elements());
        return this->data_[x];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(size_t x, size_t y) {
        size_t idx = this->calculate_offset(x, y);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(size_t x, size_t y) const {
        size_t idx = this->calculate_offset(x, y);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z) {
        size_t idx = this->calculate_offset(x, y, z);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z) const {
        size_t idx = this->calculate_offset(x, y, z);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z, size_t s) {
        size_t idx = this->calculate_offset(x, y, z, s);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z, size_t s) const {
        size_t idx = this->calculate_offset(x, y, z, s);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z, size_t s, size_t p) {
        size_t idx = this->calculate_offset(x, y, z, s, p);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z, size_t s, size_t p) const {
        size_t idx = this->calculate_offset(x, y, z, s, p);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r) {
        size_t idx = this->calculate_offset(x, y, z, s, p, r);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r) const {
        size_t idx = this->calculate_offset(x, y, z, s, p, r);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a) {
        size_t idx = this->calculate_offset(x, y, z, s, p, r, a);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(
            size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a) const {
        size_t idx = this->calculate_offset(x, y, z, s, p, r, a);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q) {
        size_t idx = this->calculate_offset(x, y, z, s, p, r, a, q);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(
            size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q) const {
        size_t idx = this->calculate_offset(x, y, z, s, p, r, a, q);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline T &hoNDArray<T>::operator()(
            size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q, size_t u) {
        size_t idx = this->calculate_offset(x, y, z, s, p, r, a, q, u);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }

    template<typename T>
    inline const T &hoNDArray<T>::operator()(
            size_t x, size_t y, size_t z, size_t s, size_t p, size_t r, size_t a, size_t q, size_t u) const {
        size_t idx = this->calculate_offset(x, y, z, s, p, r, a, q, u);
        GADGET_DEBUG_CHECK_THROW(idx < this->get_number_of_elements());
        return this->data_[idx];
    }
    #if __cplusplus > 201402L

    namespace {
        struct hondarray_detail {

            template<class... ARGS>
            static auto extract_indices(const Indexing::Slice &, const ARGS &... args) {
                return extract_indices(args...);
            }

            template<class... ARGS>
            static auto extract_indices(size_t index0, const ARGS &... args) {
                return std::array<size_t, sizeof...(ARGS) + 1>{index0, static_cast<size_t>(args)...};
            }

            template<class T, class... INDICES>
            static auto calculate_strides(const hoNDArray<T> &base, const INDICES &... indices) {
                constexpr size_t ndims = gadgetron_detail::count_slices<0, INDICES...>::value;
                auto strides = std::array<size_t, ndims>{};
                std::fill_n(strides.begin(), strides.size(), 1);
                calculate_strides_internal<ndims, 0, 0>(strides, base, indices...);
                return strides;
            }

            template<class T, class... INDICES>
            static auto calculate_dimensions(const hoNDArray<T> &base, const INDICES &... indices) {
                constexpr size_t ndims = gadgetron_detail::count_slices<0, INDICES...>::value;
                auto dims = std::array<size_t, ndims>{};
                calculate_dims_internal<ndims, 0, 0>(dims, base, indices...);
                return dims;
            }

            static size_t slice_start_index(size_t x) {
                return x;
            }

            static size_t slice_start_index(const Indexing::Slice &) {
                return 0;
            }

            template<unsigned int DIMS, unsigned int CUR_DIM, class ASSIGNEE, class OTHER>
            struct looper {
                static void assign_loop(const vector_td<size_t, DIMS> &dims, std::array<size_t, DIMS> &idx,
                                        ASSIGNEE &self, const OTHER &other) {
                    for (idx[CUR_DIM] = 0; idx[CUR_DIM] < dims[CUR_DIM]; idx[CUR_DIM]++) {
                        looper<DIMS, CUR_DIM - 1, ASSIGNEE, OTHER>::assign_loop(dims, idx, self, other);
                    }
                }
            };

            template<unsigned int DIMS, class ASSIGNEE, class OTHER>
            struct looper<DIMS, 0, ASSIGNEE, OTHER> {
                static void assign_loop(const vector_td<size_t, DIMS> &dims, std::array<size_t, DIMS> &idx,
                                        ASSIGNEE &self, const OTHER &other) {
                    for (idx[0] = 0; idx[0] < dims[0]; idx[0]++) {
                        std::apply([&](auto &&... indices) { self(indices...) = other(indices...); }, idx);
                    }
                }
            };

            template<unsigned int D>
            static bool is_contigous_data(const vector_td<size_t, D> &dimensions, const vector_td<size_t, D> &strides) {

                bool result = strides[0] == 1;
                size_t total_stride = 1;
                for (long long i = 1; i < D; i++) {
                    total_stride *= dimensions[i - 1];
                    result = result && (total_stride == strides[i]);
                }

                return result;
            }

        private:
            template<unsigned int DIMS, unsigned int CUR_VIEW_DIM, unsigned int CUR_ARRAY_DIM, class T,
                    class... INDICES>
            static void calculate_dims_internal(
                    std::array<size_t, DIMS> &dims, const hoNDArray<T> &base, const size_t &,
                    const INDICES &... indices) {
                calculate_dims_internal<DIMS, CUR_VIEW_DIM, CUR_ARRAY_DIM + 1>(dims, base, indices...);
            }

            template<unsigned int DIMS, unsigned int CUR_VIEW_DIM, unsigned int CUR_ARRAY_DIM, class T,
                    class... INDICES>
            static void calculate_dims_internal(
                    std::array<size_t, DIMS> &dims, const hoNDArray<T> &base, const Indexing::Slice &,
                    const INDICES &... indices) {
                dims[CUR_VIEW_DIM] = base.get_size(CUR_ARRAY_DIM);
                if (CUR_VIEW_DIM + 1 < DIMS)
                    calculate_dims_internal<DIMS, CUR_VIEW_DIM + 1, CUR_ARRAY_DIM + 1>(dims, base, indices...);
            }

            template<unsigned int DIMS, unsigned int CUR_STRIDE_DIM, unsigned int CUR_ARRAY_DIM, class T>
            static void calculate_dims_internal(std::array<size_t, DIMS> &strides, const hoNDArray<T> &base) {}

            template<unsigned int DIMS, unsigned int CUR_STRIDE_DIM, unsigned int CUR_ARRAY_DIM, class T,
                    class... INDICES>
            static void calculate_strides_internal(std::array<size_t, DIMS> &strides, const hoNDArray<T> &base,
                                                   const size_t &x, const INDICES &... indices) {
                strides[CUR_STRIDE_DIM] *= base.get_size(CUR_ARRAY_DIM);
                hondarray_detail::calculate_strides_internal<DIMS, CUR_STRIDE_DIM, CUR_ARRAY_DIM + 1>(strides, base,
                                                                                                      indices...);
            }

            template<unsigned int DIMS, unsigned int CUR_STRIDE_DIM, unsigned int CUR_ARRAY_DIM, class T,
                    class... INDICES>
            static void calculate_strides_internal(
                    std::array<size_t, DIMS> &strides, const hoNDArray<T> &base, const Indexing::Slice &,
                    const INDICES &... indices) {
                if (CUR_STRIDE_DIM + 1 < DIMS) {
                    strides[CUR_STRIDE_DIM + 1] = strides[CUR_STRIDE_DIM] * base.get_size(CUR_ARRAY_DIM);
                    hondarray_detail::calculate_strides_internal<DIMS, CUR_STRIDE_DIM + 1, CUR_ARRAY_DIM + 1>(strides,
                                                                                                              base,
                                                                                                              indices...);
                }
            }

            template<unsigned int DIMS, unsigned int CUR_STRIDE_DIM, unsigned int CUR_ARRAY_DIM, class T>
            static void calculate_strides_internal(std::array<size_t, DIMS> &strides, const hoNDArray<T> &base) {}


        };
    }

    template<class T>
    template<class... INDICES, class UNUSED>
    auto hoNDArray<T>::operator()(const INDICES &... indices) {

        auto strides = hondarray_detail::calculate_strides(*this, indices...);
        auto dims = hondarray_detail::calculate_dimensions(*this, indices...);

        T *offset = Core::apply([this](auto &&... indices) -> T * { return &this->operator()(indices...); },
                                std::make_tuple(hondarray_detail::slice_start_index(indices)...));

        return hoNDArrayView<T, dims.size(), gadgetron_detail::is_contiguous_index<INDICES...>::value>{strides, dims,
                                                                                                       offset};
    }

    template<typename T>
    template<class... INDICES, class>
    auto hoNDArray<T>::operator()(const INDICES &... indices) const
    -> const hoNDArrayView<T, gadgetron_detail::count_slices<0, INDICES...>::value, gadgetron_detail::is_contiguous_index<INDICES...>::value> {

        auto strides = hondarray_detail::calculate_strides(*this, indices...);
        auto dims = hondarray_detail::calculate_dimensions(*this, indices...);

        const T *offset = Core::apply([this](auto &&... indices) -> const T * { return &this->operator()(indices...); },
                                      std::make_tuple(hondarray_detail::slice_start_index(indices)...));

        return hoNDArrayView<T, dims.size(), gadgetron_detail::is_contiguous_index<INDICES...>::value>{strides, dims,
                                                                                                       const_cast<T *>(offset)};
    }

    template<typename T>
    template<unsigned int D, bool C>
    hoNDArray<T> &hoNDArray<T>::operator=(const hoNDArrayView<T, D, C> &view) {
        auto other_dims = to_std_vector(view.dimensions);
        if (!this->dimensions_equal(to_std_vector(view.dimensions))) {
            this->create(other_dims);
        }

        auto idx = std::array<size_t, D>{};
        hondarray_detail::looper<D, D - 1, hoNDArray<T>, hoNDArrayView<T, D, C>>::assign_loop(view.dimensions, idx,
                                                                                              *this, view);


    }

    template<class T, size_t D, bool C>
    hoNDArrayView<T, D, C> &hoNDArrayView<T, D, C>::operator=(const hoNDArrayView<T, D, C> &other) {
        if (&other == this) return *this;
        if (this->dimensions != other.dimensions){
            throw std::runtime_error("Dimensions must be the same for slice assignment");
        }
        if constexpr(C){
            size_t elements = std::accumulate(dimensions.begin(),dimensions.end(),size_t(1),std::multiplies<>());
            std::copy_n(other.data,elements,this->data);
            return *this;
        }
        auto idx = std::array<size_t, D>{};
        hondarray_detail::looper<D, D - 1, hoNDArrayView<T, D, C>, hoNDArrayView<T, D,C>>::assign_loop(dimensions, idx,
                                                                                                     *this, other);
        return *this;

    }
    template<class T, size_t D, bool C>
    hoNDArrayView<T, D, C> &hoNDArrayView<T, D, C>::operator=(const hoNDArrayView<T, D, !C> &other) {
        if (this->dimensions != other.dimensions){
            throw std::runtime_error("Dimensions must be the same for slice assignment");
        }
        auto idx = std::array<size_t, D>{};
        hondarray_detail::looper<D, D - 1, hoNDArrayView<T, D, C>, hoNDArrayView<T, D,!C>>::assign_loop(dimensions, idx,
                                                                                                     *this, other);
        return *this;

    }



    template<class T, size_t D, bool C>
    hoNDArrayView<T, D, C> &hoNDArrayView<T, D, C>::operator=(const hoNDArray<T> &other) {
        if (this->dimensions.size() != other.dimensions().size() && !std::equal(dimensions.begin(),dimensions.end(),other.dimensions().begin())){
            throw std::runtime_error("Dimensions must be the same for slice assignment");
        }

        auto idx = std::array<size_t, D>{};
        hondarray_detail::looper<D, D - 1, hoNDArrayView<T, D, C>, hoNDArray<T>>::assign_loop(dimensions, idx, *this,
                                                                                              other);
        return *this;
    }

    template<class T, size_t D, bool C>
    template<class... INDICES>
    std::enable_if_t<Core::all_of_v<Core::is_convertible_v<INDICES, size_t>...> && (sizeof...(INDICES) == D), T &>
    hoNDArrayView<T, D, C>::operator()(INDICES... indices) {
        auto index_vector = make_vector_td<size_t>(indices...);
        size_t offset = sum(index_vector * this->strides);
        return data[offset];
    }

    template<class T, size_t D, bool C>
    template<class... INDICES>
    std::enable_if_t<Core::all_of_v<Core::is_convertible_v<INDICES, size_t>...> && (sizeof...(INDICES) == D), const T &>
    hoNDArrayView<T, D, C>::operator()(INDICES... indices) const {
        auto index_vector = make_vector_td<size_t>(indices...);
        size_t offset = sum(index_vector * this->strides);
        return data[offset];
    }

    template<class T, size_t D, bool contiguous>
    hoNDArrayView<T, D, contiguous>::hoNDArrayView(const std::array<size_t, D> &strides,
                                                   const std::array<size_t, D> &dimensions, T *data) : strides{strides},
                                                                                                       dimensions{
                                                                                                               dimensions},
                                                                                                       data{data} {}


    template<class T, size_t D, bool contiguous>
    hoNDArrayView<T, D, contiguous>::operator const hoNDArray<T>() const {
//        if constexpr(contiguous) {
//            return hoNDArray<T>(to_std_vector(dimensions), data);
//        }

        if (hondarray_detail::is_contigous_data(dimensions, strides)) {
            return hoNDArray<T>(to_std_vector(dimensions), data);
        }

        hoNDArray<T> result = *this;

        return result;

    }

//
    template<class T, size_t D, bool contigous>
    template<typename Dummy1, typename Dummy2>
    hoNDArrayView<T, D, contigous>::operator hoNDArray<T>() {
        return hoNDArray<T>(to_std_vector(dimensions), data);
    }
    #endif

}




///////////////////////////////////////////////////////////////////////////////////////////////
///  cuNDArray
///////////////////////////////////////////////////////////////////////////////////////////////

/** \file cuNDArray.h
\brief GPU-based N-dimensional array (data container)
*/

// #include "NDArray.h"
// #include "hoNDArray.h"
// #include "complext.h"
// #include "GadgetronCuException.h"
// #include "check_CUDA.h"
// #include <boost/shared_ptr.hpp>
// #include <boost/make_shared.hpp>

// #include <cuda.h>
// #include <cuda_runtime_api.h>
// #include <thrust/device_vector.h>

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
            CUDA_CALL(
                DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(this->data_, a.data_, this->elements_ * sizeof(T))));
        } else {
            //This memory is on a different device, we must move it.
            /*
            DPCT1093:6: The "a.device_" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            dpct::select_device(a.device_);
            std::shared_ptr< hoNDArray<T> > tmp = a.to_host();
            /*
            DPCT1093:7: The "this->device_" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            dpct::select_device(this->device_);
            dpct::err0 err = DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                                  .memcpy(this->data_, tmp->get_data_ptr(), this->elements_ * sizeof(T))
                                                  .wait());
            /*
            DPCT1000:9: Error handling if-stmt was detected but could not be rewritten.
            */
            if (err != 0) {
                /*
                DPCT1001:8: The statement could not be removed.
                */
                deallocate_memory();
                this->data_ = 0;
                this->dimensions_.clear();
                throw cuda_error(err);
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
        if (DPCT_CHECK_ERROR(
                dpct::get_in_order_queue().memcpy(this->data_, a.get_data_ptr(), this->elements_ * sizeof(T)).wait()) !=
            0) {
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
        CUDA_CALL(DPCT_CHECK_ERROR(cur_device = dpct::get_current_device_id()));
        bool dimensions_match = this->dimensions_equal(rhs);
        if (dimensions_match && (rhs.device_ == cur_device) && (cur_device == this->device_)) {
            CUDA_CALL(DPCT_CHECK_ERROR(
                dpct::get_in_order_queue().memcpy(this->data_, rhs.data_, this->elements_ * sizeof(T))));
        }
        else {
            /*
            DPCT1093:10: The "this->device_" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(this->device_)));
            if( !dimensions_match ){
                deallocate_memory();
                this->elements_ = rhs.elements_;
                this->dimensions_ = rhs.dimensions_;
                allocate_memory();
            }
            if (this->device_ == rhs.device_) {
                if (DPCT_CHECK_ERROR(
                        dpct::get_in_order_queue().memcpy(this->data_, rhs.data_, this->elements_ * sizeof(T))) != 0) {
                    /*
                    DPCT1093:11: The "cur_device" device may be not the one intended for use. Adjust the selected device
                    if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: failed to copy data (2)");
                }
            } else {
                /*
                DPCT1093:12: The "rhs.device_" device may be not the one intended for use. Adjust the selected device if
                needed.
                */
                if (DPCT_CHECK_ERROR(dpct::select_device(rhs.device_)) != 0) {
                    /*
                    DPCT1093:13: The "cur_device" device may be not the one intended for use. Adjust the selected device
                    if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: unable to set device no (2)");
                }
                std::shared_ptr< hoNDArray<T> > tmp = rhs.to_host();
                /*
                DPCT1093:14: The "this->device_" device may be not the one intended for use. Adjust the selected device
                if needed.
                */
                if (DPCT_CHECK_ERROR(dpct::select_device(this->device_)) != 0) {
                    /*
                    DPCT1093:15: The "cur_device" device may be not the one intended for use. Adjust the selected device
                    if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: unable to set device no (3)");
                }
                if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(this->data_, tmp->get_data_ptr(), this->elements_ * sizeof(T))
                                         .wait()) != 0) {
                    /*
                    DPCT1093:16: The "cur_device" device may be not the one intended for use. Adjust the selected device
                    if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: failed to copy data (3)");
                }
            }
            /*
            DPCT1093:17: The "cur_device" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            if (DPCT_CHECK_ERROR(dpct::select_device(cur_device)) != 0) {
                throw cuda_error("cuNDArray::operator=: unable to restore to current device");
            }
        }
        return *this;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T> cuNDArray<T>& cuNDArray<T>::operator=(const hoNDArray<T>& rhs) try {
        int cur_device;
        CUDA_CALL(DPCT_CHECK_ERROR(cur_device = dpct::get_current_device_id()));
        bool dimensions_match = this->dimensions_equal(rhs);
        if (dimensions_match && (cur_device == this->device_)) {
            CUDA_CALL(DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memcpy(this->get_data_ptr(), rhs.get_data_ptr(), this->get_number_of_elements() * sizeof(T))
                    .wait()));
        }
        else {
            /*
            DPCT1093:18: The "this->device_" device may be not the one intended for use. Adjust the selected device if
            needed.
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
                        .memcpy(this->get_data_ptr(), rhs.get_data_ptr(), this->get_number_of_elements() * sizeof(T))
                        .wait()) != 0) {
                    /*
                    DPCT1093:19: The "cur_device" device may be not the one intended for use. Adjust the selected device
                    if needed.
                    */
                    dpct::select_device(cur_device);
                    throw cuda_error("cuNDArray::operator=: failed to copy data (1)");
            }
            /*
            DPCT1093:20: The "cur_device" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            if (DPCT_CHECK_ERROR(dpct::select_device(cur_device)) != 0) {
                throw cuda_error("cuNDArray::operator=: unable to restore to current device");
            }
        }
        return *this;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
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
    inline void cuNDArray<T>::create(const std::vector<size_t>& dimensions, T* data, bool delete_data_on_destruct) try {
        if (!data) {
            throw std::runtime_error("cuNDArray::create: 0x0 pointer provided");
        }

        int tmp_device;
        if (DPCT_CHECK_ERROR(tmp_device = dpct::get_current_device_id()) != 0) {
            throw cuda_error("cuNDArray::create: Unable to query for device");
        }

        dpct::device_info deviceProp;
        if (DPCT_CHECK_ERROR(dpct::get_device(tmp_device).get_device_info(deviceProp)) != 0) {
            throw cuda_error("cuNDArray::create: Unable to query device properties");
        }

        if (deviceProp.get_host_unified_memory()) {
            dpct::pointer_attributes attrib;
            if (cudaPointerGetAttributes(&attrib, data) != 0) {
                CHECK_FOR_CUDA_ERROR();
                throw cuda_error("cuNDArray::create: Unable to determine attributes of pointer");
            }
            this->device_ = attrib.get_device_id();
        } else {
            this->device_ = tmp_device;
        }

        Gadgetron::NDArray<T>::create(dimensions, data, delete_data_on_destruct);
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
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

        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(out->get_data_ptr(), this->data_, this->elements_ * sizeof(T))
                                 .wait()) != 0) {
            throw cuda_error("cuNDArray::to_host(): failed to copy memory from device");
        }
    }

    template <typename T> inline void cuNDArray<T>::set_device(int device) try {
        if( device_ == device )
            return;

        int cur_device;
        if (DPCT_CHECK_ERROR(cur_device = dpct::get_current_device_id()) != 0) {
            throw cuda_error("cuNDArray::set_device: unable to get device no");
        }

        /*
        DPCT1093:21: The "device_" device may be not the one intended for use. Adjust the selected device if needed.
        */
        if (cur_device != device_ && DPCT_CHECK_ERROR(dpct::select_device(device_)) != 0) {
            throw cuda_error("cuNDArray::set_device: unable to set device no");
        }

        std::shared_ptr< hoNDArray<T> > tmp = to_host();
        deallocate_memory();
        /*
        DPCT1093:22: The "device" device may be not the one intended for use. Adjust the selected device if needed.
        */
        if (DPCT_CHECK_ERROR(dpct::select_device(device)) != 0) {
            /*
            DPCT1093:23: The "cur_device" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            dpct::select_device(cur_device);
            throw cuda_error("cuNDArray::set_device: unable to set device no (2)");
        }

        device_ = device;
        allocate_memory();
        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(this->data_, tmp->get_data_ptr(), this->elements_ * sizeof(T))
                                 .wait()) != 0) {
            /*
            DPCT1093:24: The "cur_device" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            dpct::select_device(cur_device);
            throw cuda_error("cuNDArray::set_device: failed to copy data");
        }

        /*
        DPCT1093:25: The "cur_device" device may be not the one intended for use. Adjust the selected device if needed.
        */
        if (DPCT_CHECK_ERROR(dpct::select_device(cur_device)) != 0) {
            throw cuda_error("cuNDArray::set_device: unable to restore device to current device");
        }
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
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
        CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(&res, &this->get_data_ptr()[idx], sizeof(T)).wait()));
        return res;
    }

    template <typename T> 
    inline T cuNDArray<T>::operator[]( size_t idx )
    {
        if( idx >= this->get_number_of_elements() ){
            throw std::runtime_error("cuNDArray::operator[]: index out of range.");
        }
        T res;
        CUDA_CALL(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(&res, &this->get_data_ptr()[idx], sizeof(T)).wait()));
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
            DPCT1093:26: The "device_" device may be not the one intended for use. Adjust the selected device if needed.
            */
            if (DPCT_CHECK_ERROR(dpct::select_device(device_)) != 0) {
                throw cuda_error("cuNDArray::allocate_memory: unable to set device no");
            }
        }

        if (DPCT_CHECK_ERROR((this->data_) = (typename std::remove_reference<decltype(this->data_)>::type)
                                 sycl::malloc_device(size, dpct::get_in_order_queue())) != 0) {
            size_t free = 0, total = 0;
            /*
            DPCT1106:27: 'cudaMemGetInfo' was migrated with the Intel extensions for device information which may not be
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
            DPCT1093:28: The "device_no_old" device may be not the one intended for use. Adjust the selected device if
            needed.
            */
            if (DPCT_CHECK_ERROR(dpct::select_device(device_no_old)) != 0) {
                throw cuda_error("cuNDArray::allocate_memory: unable to restore device no");
            }
        }
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    template <typename T> void cuNDArray<T>::deallocate_memory() try {
        if (this->data_) {

            int device_no_old;
            CUDA_CALL(DPCT_CHECK_ERROR(device_no_old = dpct::get_current_device_id()));
            if (device_ != device_no_old) {
                /*
                DPCT1093:29: The "device_" device may be not the one intended for use. Adjust the selected device if
                needed.
                */
                CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(device_)));
            }

            CUDA_CALL(DPCT_CHECK_ERROR(dpct::dpct_free(this->data_, dpct::get_in_order_queue())));
            if (device_ != device_no_old) {
                /*
                DPCT1093:30: The "device_no_old" device may be not the one intended for use. Adjust the selected device
                if needed.
                */
                CUDA_CALL(DPCT_CHECK_ERROR(dpct::select_device(device_no_old)));
            }
            this->data_ = 0;
        }
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
}




