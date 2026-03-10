#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuNDArray_operators.h"
#include "complext.h"
#include <functional>

#include <dpct/dpl_utils.hpp>

#include <complex>

using namespace Gadgetron;
  // Private utility to verify array dimensions. 
  // It "replaces" NDArray::dimensions_equal() to support batch mode.
  // There is an identical function for all array instances (currently hoNDArray, cuNDArray, hoCuNDAraay)
  // !!! Remember to fix any bugs in all versions !!!
  //
  template<class T,class S> static bool compatible_dimensions( const cuNDArray<T> &x, const cuNDArray<S> &y )
  {
    return ((x.get_number_of_elements()%y.get_number_of_elements())==0);
  }

  template <typename T>
  /*
  DPCT1044:31: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need
  to remove references to typedefs from thrust::unary_function in the class definition.
  */
  class cuNDA_modulus {
  public:
    cuNDA_modulus(int x):mod(x) {};
    T operator()(const T &y) const {return y%mod;}
  private:
    const int mod;
  };

  //
  // This transform support batch mode when the number of elements in x is a multiple of the number of elements in y
  //
  template<class T,class S,class F>  
  static void equals_transform(cuNDArray<T> &x, const cuNDArray<S> &y){
    if (x.dimensions_equal(y)){
      std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(),
                     y.begin(), x.begin(), F());
    } else if (compatible_dimensions(x,y))
      {
        if (y.get_number_of_elements() < x.get_number_of_elements()) {
          typedef oneapi::dpl::transform_iterator<cuNDA_modulus<int>, oneapi::dpl::counting_iterator<int>, int>
              transform_it;
          transform_it indices = oneapi::dpl::make_transform_iterator(dpct::make_counting_iterator(0),
                                                                      cuNDA_modulus<int>(y.get_number_of_elements()));
          oneapi::dpl::permutation_iterator<dpct::device_pointer<S>, transform_it> p =
              oneapi::dpl::make_permutation_iterator(y.begin(), indices);
          std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), p,
                         x.begin(), F());
        } else {
          std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(),
                         y.begin(), x.begin(), F());
        }

      } else {
      throw std::runtime_error("The provided cuNDArrays have incompatible dimensions for Gadgetron::operator {+=,-=,*=,/=}");
    }
  }

  template <typename T>
  /*
  DPCT1044:32: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::binary_function in the class definition.
  */
  struct cuNDA_plus {
    complext<T> operator()(const complext<T> &x, const T &y) const {return x+y;}
  };

  template <typename T>
  /*
  DPCT1044:33: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::binary_function in the class definition.
  */
  struct cuNDA_minus {
    complext<T> operator()(const complext<T> &x, const T &y) const {return x-y;}
  };

  template <typename T>
  /*
  DPCT1044:34: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::binary_function in the class definition.
  */
  struct cuNDA_multiply {
    complext<T> operator()(const complext<T> &x, const T &y) const {return x*y;}
  };

  template <typename T>
  /*
  DPCT1044:35: thrust::binary_function was removed because std::binary_function has been deprecated in C++11. You may
  need to remove references to typedefs from thrust::binary_function in the class definition.
  */
  struct cuNDA_divide {
    complext<T> operator()(const complext<T> &x, const T &y) const {return x/y;}
  };

  template<class T, class> cuNDArray<T> & Gadgetron::operator+= (cuNDArray<T> &x, const  cuNDArray<T> &y){
    equals_transform<T, T, std::plus<T>>(x, y);
    return x;
  }

  template<class T, class> cuNDArray<T > & Gadgetron::operator+= (cuNDArray<T> &x , T y){
    dpct::constant_iterator<T> iter(y);
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), std::plus<T>());
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator+= (cuNDArray< complext<T> > &x , const cuNDArray<T> &y){
    equals_transform< complext<T>,T,cuNDA_plus<T> >(x,y);
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator+= (cuNDArray<complext<T> > &x , T y){
    dpct::constant_iterator<T> iter(y);
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), cuNDA_plus<T>());
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator-= (cuNDArray<T> & x , const cuNDArray<T> & y){
    equals_transform<T, T, std::minus<T>>(x, y);
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator-= (cuNDArray<T> &x , T y){
    dpct::constant_iterator<T> iter(y);
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), std::minus<T>());
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator-= (cuNDArray< complext<T> > &x , const cuNDArray<T> &y){
    equals_transform< complext<T>,T,cuNDA_minus<T> >(x,y);
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator-= (cuNDArray<complext<T> > &x , T y){
    dpct::constant_iterator<T> iter(y);
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), cuNDA_minus<T>());
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator*= (cuNDArray<T> &x , const cuNDArray<T> &y){
    equals_transform<T, T, std::multiplies<T>>(x, y);
    return x;
  }

  template<class T, class> cuNDArray<T>& Gadgetron::operator*= (cuNDArray<T> &x , T y){
    dpct::constant_iterator<T> iter(y);
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), std::multiplies<T>());
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator*= (cuNDArray< complext<T> > &x , const cuNDArray<T> &y){
    equals_transform< complext<T>,T,cuNDA_multiply<T> >(x,y);
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator*= (cuNDArray<complext<T> > &x , T y){
    dpct::constant_iterator<T> iter(y);
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), cuNDA_multiply<T>());
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator/= (cuNDArray<T> &x , const cuNDArray<T> &y){
    equals_transform<T, T, std::divides<T>>(x, y);
    return x;
  }

  template<class T, class> cuNDArray<T >& Gadgetron::operator/= (cuNDArray<T> &x , T y){
    dpct::constant_iterator<T> iter(y);
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), std::divides<T>());
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator/= (cuNDArray< complext<T> > &x , const cuNDArray<T> &y){
    equals_transform< complext<T>,T,cuNDA_divide<T> >(x,y);
    return x;
  }

  template<class T, class> cuNDArray<complext<T > >& Gadgetron::operator/= (cuNDArray<complext<T> > &x , T y){
    dpct::constant_iterator<T> iter(y);
    std::transform(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), x.begin(), x.end(), iter,
                   x.begin(), cuNDA_divide<T>());
    return x;
  }


  cuNDArray<bool>& Gadgetron::operator&= (cuNDArray<bool> &x , cuNDArray<bool> &y){

    equals_transform<bool, bool, std::logical_and<bool>>(x, y);
    return x;
  }
  cuNDArray<bool>& Gadgetron::operator|= (cuNDArray<bool> &x , cuNDArray<bool> &y){

    equals_transform<bool, bool, std::logical_or<bool>>(x, y);
    return x;
  }

  //
  // Instantiation
  //
  template cuNDArray<float>& Gadgetron::operator+=<float>(cuNDArray<float>& x, const cuNDArray<float>& y);
  template cuNDArray<float>& Gadgetron::operator+=<float>(cuNDArray<float>& x, float y);
  template cuNDArray<complext<float>>& Gadgetron::operator+=<float>(cuNDArray<complext<float>>& x, const cuNDArray<float>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator+=<float>(cuNDArray<complext<float>>& x, float y);
  template cuNDArray<float>& Gadgetron::operator-=(cuNDArray<float>& x, const cuNDArray<float>& y);
  template cuNDArray<float>& Gadgetron::operator-=(cuNDArray<float>& x, float y);
  template cuNDArray<complext<float>>& Gadgetron::operator-=<float>(cuNDArray<complext<float>>& x, const cuNDArray<float>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator-=<float>(cuNDArray<complext<float>>& x, float y);
  template cuNDArray<float>& Gadgetron::operator*=<float>(cuNDArray<float>& x, const cuNDArray<float>& y);
  template cuNDArray<float>& Gadgetron::operator*=<float>(cuNDArray<float>& x, float y);
  template cuNDArray<complext<float>>& Gadgetron::operator*=<float>(cuNDArray<complext<float>>& x, const cuNDArray<float>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator*=<float>(cuNDArray<complext<float>>& x, float y);
  template cuNDArray<float>& Gadgetron::operator/=<float>(cuNDArray<float>& x, const cuNDArray<float>& y);
  template cuNDArray<float>& Gadgetron::operator/=<float>(cuNDArray<float>& x, float y);
  template cuNDArray<complext<float>>& Gadgetron::operator/=<float>(cuNDArray<complext<float>>& x,const cuNDArray<float>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator/=<float>(cuNDArray<complext<float>>& x, float y);

  template cuNDArray<double>& Gadgetron::operator+=<double>(cuNDArray<double>& x, const cuNDArray<double>& y);
  template cuNDArray<double>& Gadgetron::operator+=<double>(cuNDArray<double>& x, double y);
  template cuNDArray<complext<double>>& Gadgetron::operator+=<double>(cuNDArray<complext<double>>& x, const cuNDArray<double>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator+=<double>(cuNDArray<complext<double>>& x, double y);
  template cuNDArray<double>& Gadgetron::operator-=(cuNDArray<double>& x, const cuNDArray<double>& y);
  template cuNDArray<double>& Gadgetron::operator-=(cuNDArray<double>& x, double y);
  template cuNDArray<complext<double>>& Gadgetron::operator-=<double>(cuNDArray<complext<double>>& x, const cuNDArray<double>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator-=<double>(cuNDArray<complext<double>>& x, double y);
  template cuNDArray<double>& Gadgetron::operator*=<double>(cuNDArray<double>& x, const cuNDArray<double>& y);
  template cuNDArray<double>& Gadgetron::operator*=<double>(cuNDArray<double>& x, double y);
  template cuNDArray<complext<double>>& Gadgetron::operator*=<double>(cuNDArray<complext<double>>& x, const cuNDArray<double>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator*=<double>(cuNDArray<complext<double>>& x, double y);
  template cuNDArray<double>& Gadgetron::operator/=<double>(cuNDArray<double>& x, const cuNDArray<double>& y);
  template cuNDArray<double>& Gadgetron::operator/=<double>(cuNDArray<double>& x, double y);
  template cuNDArray<complext<double>>& Gadgetron::operator/=<double>(cuNDArray<complext<double>>& x,const cuNDArray<double>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator/=<double>(cuNDArray<complext<double>>& x, double y);


  template cuNDArray<complext<float>>& Gadgetron::operator+=<complext<float>>(cuNDArray<complext<float>>& x, const cuNDArray<complext<float>>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator+=<complext<float>>(cuNDArray<complext<float>>& x, complext<float> y);
  template cuNDArray<complext<float>>& Gadgetron::operator-=(cuNDArray<complext<float>>& x, const cuNDArray<complext<float>>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator-=(cuNDArray<complext<float>>& x, complext<float> y);
  template cuNDArray<complext<float>>& Gadgetron::operator*=<complext<float>>(cuNDArray<complext<float>>& x, const cuNDArray<complext<float>>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator*=<complext<float>>(cuNDArray<complext<float>>& x, complext<float> y);
  template cuNDArray<complext<float>>& Gadgetron::operator/=<complext<float>>(cuNDArray<complext<float>>& x, const cuNDArray<complext<float>>& y);
  template cuNDArray<complext<float>>& Gadgetron::operator/=<complext<float>>(cuNDArray<complext<float>>& x, complext<float> y);

template cuNDArray<complext<double>>& Gadgetron::operator+=<complext<double>>(cuNDArray<complext<double>>& x, const cuNDArray<complext<double>>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator+=<complext<double>>(cuNDArray<complext<double>>& x, complext<double> y);
  template cuNDArray<complext<double>>& Gadgetron::operator-=(cuNDArray<complext<double>>& x, const cuNDArray<complext<double>>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator-=(cuNDArray<complext<double>>& x, complext<double> y);
  template cuNDArray<complext<double>>& Gadgetron::operator*=<complext<double>>(cuNDArray<complext<double>>& x, const cuNDArray<complext<double>>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator*=<complext<double>>(cuNDArray<complext<double>>& x, complext<double> y);
  template cuNDArray<complext<double>>& Gadgetron::operator/=<complext<double>>(cuNDArray<complext<double>>& x, const cuNDArray<complext<double>>& y);
  template cuNDArray<complext<double>>& Gadgetron::operator/=<complext<double>>(cuNDArray<complext<double>>& x, complext<double> y);



  cuNDArray<bool>& Gadgetron::operator&=(cuNDArray<bool>& x, cuNDArray<bool>& y);
  cuNDArray<bool>& Gadgetron::operator|=(cuNDArray<bool>& x, cuNDArray<bool>& y);
  
