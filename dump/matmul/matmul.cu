// A: [M, K]
// B: typically [K, N], but to match pytorch [N, K]
// C: [M, N]

// TODO: support randn init

#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuda_bf16.h>


constexpr int default_precision = 5;

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

uint32_t ceil_div_uint32(uint32_t x, uint32_t y) {
  return (x + y - 1) / y;
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
  // A [33, 33]
  // B [33, 33]
  // ceil(33, 32) = 2
  // 64 * 64
  dim3 grid(ceil_div_uint32(m, block.x), ceil_div_uint32(n, block.n));
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
      std::cout << __bfloat162float(a[i * cols + j]) << " \n"[j == cols - 1];
    }
  }
}

void test_correctness(const std::vector<impl_t> &impls) {
  uint32_t shapes[][3] = {
    {3, 3, 3},
    {17, 13, 16},
    {17, 13, 23},
    {16, 33, 257},
    {257, 257, 257},
  };
  for (auto &[m, n, k]: shapes) {
    bf16 *a = new bf16[m * k];
    bf16 *b = new bf16[n * k];
    bf16 *c = new bf16[m * n];
    matmul_t mm = {a, b, c, m, n, k};
    init_constant(mm.a, m * k, 1.0f);
    init_constant(mm.b, n * k, 1.0f);
    init_constant(mm.c, m * n, 0.0f);
    matmul_cpu_reference(&mm);
    for (auto &impl: impls) {
      // TODO: compute outputs of all implementations and compare against cpu reference
    }
    // print_matrix(mm.c, m, n);
    delete[] mm.a;
    delete[] mm.b;
    delete[] mm.c;
  }
}


int main() {
  std::cout << std::setprecision(default_precision);

  std::vector<impl_t> impls = {
    {"matmul_v1", launch_matmul_v1};
  }
  test_correctness(impls);

  return 0;
}
