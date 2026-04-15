// A: [M, K]
// B: typically [K, N], but to match pytorch [N, K]
// C: [M, N]

#include <iostream>

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

void init_constant(bf16 *const a, uint32_t size, float value) {
  bf16 v = __float2bfloat16(value);
  for (uint32_t i = 0; i < size; ++i) {
    a[i] = v;
  }
}

void matmul_cpu_reference(matmul_t *const mm) {
  for (uint32_t i = 0; i < mm->m; ++i) {
    for (uint32_t j = 0; j < mm->n; ++j) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < mm->k; ++k) {
        sum += __bfloat162float(mm->a[i * mm->k + k]) * __bfloat162float(mm->b[j * mm->k + k]);
      }
      mm->c[i * n + j] = __float2bfloat16(sum);
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

void test_correctness() {
  uint32_t shapes[][3] = {
    {3, 3, 3}
  };
  for (auto &[m, n, k]: shapes) {
    matmul_t mm;
    init_constant(mm->a, m * k, 1.0f);
    init_constant(mm->b, n * k, 1.0f);
    init_constant(mm->c, m * n, 0.0f);
    matmul_cpu_reference(&mm);
    print_matrix(mm->c, m, n);
  }
}


int main() {
  std::cout << std::setprecision(default_precision);

  std::cout << "hello world" << '\n';
  test_correctness();

  return 0;
}
