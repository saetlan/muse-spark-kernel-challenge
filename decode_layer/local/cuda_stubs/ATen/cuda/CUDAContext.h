#pragma once
#include "../../cuda_stub_prelude.h"
#include <string>
#include <vector>
namespace at { enum class ScalarType { Int, Float, BFloat16 }; constexpr ScalarType kInt = ScalarType::Int; }
namespace torch {
struct Tensor {
  void* data_ptr() const;
  int64_t numel() const;
  int64_t size(int64_t dim) const;
  int64_t element_size() const;
  at::ScalarType scalar_type() const;
  bool is_contiguous() const;
  template <class T> T* data_ptr() const;
  Tensor view(std::initializer_list<int64_t> shape) const;
};
Tensor tensor(std::initializer_list<int64_t> values);
Tensor tensor(const std::vector<int64_t>& values);
}
namespace at { namespace cuda { struct CUDAStream { operator cudaStream_t() const; }; CUDAStream getCurrentCUDAStream(); } }
#define TORCH_CHECK(cond, ...) do { if (!(cond)) throw 0; } while (0)
