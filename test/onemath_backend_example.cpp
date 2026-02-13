// Example: oneMath compile-time backend dispatch
// Demonstrates using cublas backend on NVIDIA GPU for BLAS operations.
//
// Compile (inside container):
//   /dpcpp_home/install/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
//     -Xsycl-target-backend --cuda-gpu-arch=sm_80 \
//     -I/oneMKLwithCublas/include \
//     -L/oneMKLwithCublas/lib \
//     -lonemath_blas_cublas -lsycl -lcublas \
//     -o onemath_backend_example onemath_backend_example.cpp

#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>
#include <oneapi/math/blas.hpp>
#include <iostream>
#include <vector>

int main() {
    // --- Device & queue setup ---
    sycl::device gpu_dev;
    try {
        gpu_dev = sycl::device(sycl::gpu_selector_v);
        std::cout << "GPU: " << gpu_dev.get_info<sycl::info::device::name>() << std::endl;
    } catch (sycl::exception const& e) {
        std::cerr << "No GPU found: " << e.what() << std::endl;
        return 1;
    }

    sycl::queue gpu_queue(gpu_dev);

    // --- GEMM: C = alpha * A * B + beta * C ---
    // A: m x k, B: k x n, C: m x n (column-major)
    constexpr std::int64_t m = 4, n = 4, k = 4;
    constexpr float alpha = 1.0f, beta = 0.0f;
    constexpr std::int64_t lda = m, ldb = k, ldc = m;

    // Host data
    std::vector<float> h_A(m * k, 1.0f);  // all ones
    std::vector<float> h_B(k * n, 2.0f);  // all twos
    std::vector<float> h_C(m * n, 0.0f);

    // Device allocations (USM)
    float* d_A = sycl::malloc_device<float>(m * k, gpu_queue);
    float* d_B = sycl::malloc_device<float>(k * n, gpu_queue);
    float* d_C = sycl::malloc_device<float>(m * n, gpu_queue);

    // Copy to device
    gpu_queue.memcpy(d_A, h_A.data(), m * k * sizeof(float)).wait();
    gpu_queue.memcpy(d_B, h_B.data(), k * n * sizeof(float)).wait();
    gpu_queue.memcpy(d_C, h_C.data(), m * n * sizeof(float)).wait();

    // --- Compile-time backend dispatch: cublas ---
    oneapi::math::backend_selector<oneapi::math::backend::cublas> cublas_selector(gpu_queue);

    oneapi::math::blas::column_major::gemm(
        cublas_selector,
        oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans,
        m, n, k,
        alpha,
        d_A, lda,
        d_B, ldb,
        beta,
        d_C, ldc);

    gpu_queue.wait();

    // Copy result back
    gpu_queue.memcpy(h_C.data(), d_C, m * n * sizeof(float)).wait();

    // Print result - each element should be k * 1.0 * 2.0 = 8.0
    std::cout << "C[0] = " << h_C[0] << " (expected " << k * 1.0f * 2.0f << ")" << std::endl;

    // --- Also works with inline selector ---
    oneapi::math::blas::column_major::gemm(
        oneapi::math::backend_selector<oneapi::math::backend::cublas>{gpu_queue},
        oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans,
        m, n, k,
        alpha,
        d_A, lda,
        d_B, ldb,
        beta,
        d_C, ldc);

    gpu_queue.wait();

    // --- Example: iamax via cublas backend ---
    std::vector<float> h_x = {1.0f, -5.0f, 3.0f, -2.0f};
    float* d_x = sycl::malloc_device<float>(4, gpu_queue);
    std::int64_t* d_result = sycl::malloc_device<std::int64_t>(1, gpu_queue);
    gpu_queue.memcpy(d_x, h_x.data(), 4 * sizeof(float)).wait();

    oneapi::math::blas::column_major::iamax(
        oneapi::math::backend_selector<oneapi::math::backend::cublas>{gpu_queue},
        4,          // n
        d_x, 1,    // x, incx
        d_result);

    gpu_queue.wait();

    std::int64_t h_result;
    gpu_queue.memcpy(&h_result, d_result, sizeof(std::int64_t)).wait();
    std::cout << "iamax index = " << h_result << " (expected 1, 0-based index of -5.0)" << std::endl;

    // Cleanup
    sycl::free(d_A, gpu_queue);
    sycl::free(d_B, gpu_queue);
    sycl::free(d_C, gpu_queue);
    sycl::free(d_x, gpu_queue);
    sycl::free(d_result, gpu_queue);

    std::cout << "Done." << std::endl;
    return 0;
}
