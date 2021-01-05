#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void KeVmat(const T* rank, T* y, const int total_num, const int k1, const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    int ii = i / k1;
    y[ii * total_num + int(rank[i])] = static_cast<T>(1.0);
  }
}

// 前向OP的kernel的GPU实现
template <typename DeviceContext, typename T>
class VmatCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* initial_rank = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto x = initial_rank->data<T>();
    auto y = out_t->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    const int total_num = initial_rank->dims()[0];
    const int k1 = initial_rank->dims()[1];


    int num = initial_rank->numel();
    int block = 512;
    int grid = (num + block - 1) / block;
    KeVmat<T><<<grid, block, 0, dev_ctx.stream()>>>(x, y, total_num, k1, num);
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
// 注册前向的GPU Kernel
REGISTER_OP_CUDA_KERNEL(vmat,
                        paddle::operators::VmatCUDAKernel<CUDA, float>,
                        paddle::operators::VmatCUDAKernel<CUDA, double>);
