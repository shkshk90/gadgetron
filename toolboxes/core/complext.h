/** \file complext.h
    \brief An implementation of complex numbers that works for both the cpu and gpu.

    complext.h provides an implementation of complex numbers that, unlike std::complex,
    works on both the cpu and gpu. 
    It follows the interface defined for std::complex.
*/

#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "core_defines.h"

#include <complex>
#include <cmath>
#include <iostream>

namespace Gadgetron {

    // Scalar math wrappers that work on both host and device.
    // std:: versions are always available; on device, sycl:: builtins are also
    // pulled in so that unqualified calls from device code resolve correctly.
    using std::abs;
    using std::sin;
    using std::cos;
    using std::exp;
    using std::sqrt;
    using std::atan2;
    using std::cosh;
    using std::sinh;
#ifdef __SYCL_DEVICE_ONLY__
    using sycl::sin;
    using sycl::cos;
    using sycl::exp;
    using sycl::sqrt;
    using sycl::atan2;
    using sycl::cosh;
    using sycl::sinh;
#endif

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

        __inline__ complext(const complext<T> &tmp) = default;

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
        return complext<T>(rho * cos(theta), rho * sin(theta));
    }

    template<class T>
    __inline__ complext<T> sqrt(complext<T> x) {
        T r = abs(x);
        return complext<T>(sqrt((r + x.real()) / 2), sgn(x.imag()) * sqrt((r - x.real()) / 2));
    }

    template<class T>
    __inline__ T abs(complext<T> comp) {
        return sqrt(comp._real * comp._real + comp._imag * comp._imag);
    }

    template<class T>
    __inline__ complext<T> sin(complext<T> comp) {
        return complext<T>(sin(comp._real) * cosh(comp._imag), cos(comp._real) * sinh(comp._imag));
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
