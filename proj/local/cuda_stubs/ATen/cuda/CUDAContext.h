#pragma once
#include "../../cuda_stub_prelude.h"
#include <string>
#include <vector>
namespace torch {
struct Tensor {
  void* data_ptr() const;
  template <class T> T* data_ptr() const;
  Tensor view(std::initializer_list<int64_t> shape) const;
  int64_t numel() const;  // added for proj F12V's launch checks
  int64_t element_size() const;  // added for proj F12M's launch checks
  int64_t size(int64_t dim) const;  // added for proj WG
  bool is_contiguous() const;
  int scalar_type() const;
  int64_t dim() const;  // added for proj F12K
};
Tensor tensor(std::initializer_list<int64_t> values);
Tensor tensor(const std::vector<int64_t>& values);
}
namespace at { struct ScalarType { static constexpr int BFloat16 = 15; static constexpr int Float = 6; }; constexpr int kBFloat16 = 15; constexpr int kFloat = 6; constexpr int kInt = 3;  /* added for proj F12K */ namespace cuda { struct CUDAStream { operator cudaStream_t() const; }; CUDAStream getCurrentCUDAStream(); } }
#define TORCH_CHECK(cond, ...) do { if (!(cond)) throw 0; } while (0)
