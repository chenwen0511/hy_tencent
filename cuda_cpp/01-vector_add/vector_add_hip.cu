// 海光 DCU / ROCm：HIP 运行时，不依赖 cuda_runtime.h 与 nvcc。
// 编译：hipcc -O2 -std=c++14 -o vector_add_hip vector_add_hip.cu

#include <cmath>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

#define CHECK_HIP(call)                                                        \
    do {                                                                       \
        hipError_t err__ = (call);                                           \
        if (err__ != hipSuccess) {                                           \
            std::cerr << "HIP 调用失败: " << #call                            \
                      << " | 错误: " << hipGetErrorString(err__)             \
                      << " | 行号: " << __LINE__ << std::endl;                \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 100000;
    size_t size = numElements * sizeof(float);

    int rtVer = 0;
    CHECK_HIP(hipRuntimeGetVersion(&rtVer));
    std::cout << "HIP Runtime 版本: " << rtVer << std::endl;

    CHECK_HIP(hipSetDevice(0));

    hipDeviceProp_t prop{};
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::cout << "当前设备: " << prop.name;
#ifdef HIP_VERSION_MAJOR
    std::cout << " | HIP " << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR;
#endif
    std::cout << std::endl;

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host 内存分配失败！" << std::endl;
        free(h_A);
        free(h_B);
        free(h_C);
        return 1;
    }
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_HIP(hipMalloc((void **)&d_A, size));
    CHECK_HIP(hipMalloc((void **)&d_B, size));
    CHECK_HIP(hipMalloc((void **)&d_C, size));

    CHECK_HIP(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid =
        (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(h_C[i] - 3.0f) > 1e-6f) {
            success = false;
            break;
        }
    }
    if (success)
        std::cout << "算子执行成功！所有结果均为 3.0（HIP 路径）" << std::endl;
    else
        std::cout << "算子执行失败！" << std::endl;

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
