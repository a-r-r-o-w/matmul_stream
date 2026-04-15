// A: [M, K]
// B: typically [K, N], but to match pytorch [N, K]
// C: [M, N]

// TODO: support randn init

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>


namespace pantheos {

using bf16 = __nv_bfloat16;

}  // pantheos

using namespace pantheos;


struct matmul_t {
  bf16 *a;
  bf16 *b;
  bf16 *c;
  uint32_t m;
  uint32_t n;
  uint32_t k;
};

struct impl_t {
  using launch_fn = void(*)(matmul_t *const, uint32_t, uint32_t, uint32_t, cudaStream_t);

  const char *name;
  launch_fn fn;
};

struct statistics_t {
  float absmax = 0.0f;
  float l1_norm = 0.0f;
  float l2_norm = 0.0f;
};


uint32_t ceil_div_uint32(uint32_t x, uint32_t y) {
  return (x + y - 1) / y;
}

statistics_t compare_results(const bf16 *const x, const bf16 *const y, uint32_t size) {
  statistics_t stats;
  for (uint32_t i = 0; i < size; ++i) {
    float diff = __bfloat162float(x[i]) - __bfloat162float(y[i]);
    float absdiff = std::abs(diff);
    stats.absmax = std::max(stats.absmax, absdiff);
    stats.l1_norm += absdiff;
    stats.l2_norm += diff * diff;
  }
  stats.l1_norm /= size;
  stats.l2_norm /= size;
  return stats;
}


__global__ void matmul_v1(matmul_t *const mm) {
  auto &[a, b, c, m, n, k] = *mm;
  const uint32_t thread_row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t thread_col = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_row >= m || thread_col >= n)
    return;
  float sum = 0.0f;
  for (uint32_t i = 0; i < k; ++i) {
    sum += __bfloat162float(a[thread_row * k + i]) * __bfloat162float(b[thread_col * k + i]);
  }
  c[thread_row * n + thread_col] = sum;
}

void launch_matmul_v1(matmul_t *const mm, uint32_t m, uint32_t n, uint32_t k, cudaStream_t stream) {
  constexpr dim3 block(32, 32);
  dim3 grid(ceil_div_uint32(m, block.x), ceil_div_uint32(n, block.y));
  matmul_v1<<<grid, block, 0, stream>>>(mm);
}


void init_constant(bf16 *const a, uint32_t size, float value) {
  bf16 v = __float2bfloat16(value);
  for (uint32_t i = 0; i < size; ++i) {
    a[i] = v;
  }
}

void matmul_cpu_reference(matmul_t *const mm) {
  auto &[a, b, c, m, n, k] = *mm;
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (uint32_t v = 0; v < k; ++v) {
        sum += __bfloat162float(a[i * k + v]) * __bfloat162float(b[j * k + v]);
      }
      c[i * n + j] = __float2bfloat16(sum);
    }
  }
}

void print_matrix(bf16 *a, uint32_t rows, uint32_t cols) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      printf("%f%c", __bfloat162float(a[i * cols + j]), " \n"[j == cols - 1]);
    }
  }
}

void test_correctness(const std::vector<impl_t> &impls, cudaStream_t stream) {
  uint32_t shapes[][3] = {
    {3, 3, 3},
    {17, 13, 16},
    {17, 13, 23},
    {16, 33, 257},
    {257, 257, 257},
  };
  for (auto &[m, n, k]: shapes) {
    printf("===== shape: [m=%d, n=%d, k=%d] =====\n", m, n, k);
    bf16 *a_cpu = new bf16[m * k];
    bf16 *b_cpu = new bf16[n * k];
    bf16 *c_cpu = new bf16[m * n];
    bf16 *c_out_cpu = new bf16[m * n];
    matmul_t mm_cpu = {a_cpu, b_cpu, c_cpu, m, n, k};
    init_constant(mm_cpu.a, m * k, 1.0f);
    init_constant(mm_cpu.b, n * k, 1.0f);
    init_constant(mm_cpu.c, m * n, 0.0f);
    matmul_cpu_reference(&mm_cpu);
    printf("cpu reference"); fflush(stdout);

    bf16 *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, m * k * sizeof(bf16));
    cudaMalloc(&b_d, n * k * sizeof(bf16));
    cudaMalloc(&c_d, m * n * sizeof(bf16));
    printf("cuda malloc done"); fflush(stdout);
    cudaMemcpy(a_d, a_cpu, m * k * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_cpu, n * k * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_cpu, m * n * sizeof(bf16), cudaMemcpyHostToDevice);
    printf("memcpy"); fflush(stdout);
    matmul_t mm_d_host = {a_d, b_d, c_d, m, n, k};
    matmul_t *mm_d;
    cudaMalloc(&mm_d, sizeof(matmul_t));
    cudaMemcpy(mm_d, &mm_d_host, sizeof(matmul_t), cudaMemcpyHostToDevice);

    for (auto &impl: impls) {
      printf("here 1");
      fflush(stdout);
      impl.fn(mm_d, m, n, k, stream);
      printf("here 2"); fflush(stdout);
      cudaMemcpy(c_out_cpu, c_d, m * n * sizeof(bf16), cudaMemcpyDeviceToHost);
      printf("here 3"); fflush(stdout);
      statistics_t stats = compare_results(c_cpu, c_out_cpu, m * n);
      printf("here 4"); fflush(stdout);
      printf("[%s] absmax=%.3f l1_norm=%.3f l2_norm=%.3f\n", impl.name, stats.absmax, stats.l1_norm, stats.l2_norm);
    }
    printf("\n");

    cudaFree(mm_d);
    cudaFree(c_d);
    cudaFree(b_d);
    cudaFree(a_d);
    delete[] c_out_cpu;
    delete[] c_cpu;
    delete[] b_cpu;
    delete[] a_cpu;
  }
}


int main() {
  std::vector<impl_t> impls = {
    {"matmul_v1", launch_matmul_v1},
  };

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  test_correctness(impls, stream);

  cudaStreamDestroy(stream);

  return 0;
}
