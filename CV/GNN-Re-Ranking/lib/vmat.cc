#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class VmatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddOutput("Y", "Output of Vmat_op");
    AddComment(R"DOC(
Build neighbor relationship for rerank
)DOC");
  }
};

class VmatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    std::vector<int64_t> dim_y({in_dims[0], in_dims[0]});
    ctx->SetOutputDim("Y", framework::make_ddim(dim_y));
  }
};

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class VmatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* initial_rank = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto x = initial_rank->data<T>();
    // mutable_data分配内存、获取指针
    auto y = out_t->mutable_data<T>(ctx.GetPlace());

    const int total_num = initial_rank->dims()[0];
    const int k1 = initial_rank->dims()[1];

    for (int i = 0; i < initial_rank->numel(); ++i) {
      int ii = i / k1;
      y[ii*total_num + int(x[i])] = static_cast<T>(1.0);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
// 注册前向和反向op
REGISTER_OPERATOR(vmat,
                  ops::VmatOp,
                  ops::VmatOpMaker,
                  paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
                  paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
// 注册CPU的Kernel
REGISTER_OP_CPU_KERNEL(vmat,
                       ops::VmatKernel<CPU, float>,
                       ops::VmatKernel<CPU, double>);
