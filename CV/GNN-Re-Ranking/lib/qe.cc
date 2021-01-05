#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class QeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("V", "The Vmat.");
    AddInput("R", "The Rank.");
    AddInput("S", "The Similarity.");
    AddOutput("Y", "Output of query expand");
    AddComment(R"DOC(
Query Expansion
)DOC");
  }
};

class QeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("V");
    std::vector<int64_t> dim_y({in_dims[0], in_dims[0]});
    ctx->SetOutputDim("Y", framework::make_ddim(dim_y));
  }
};

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class QeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* V = ctx.Input<Tensor>("R");
    auto* R = ctx.Input<Tensor>("V");
    auto* S = ctx.Input<Tensor>("S");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto v = V->data<T>();
    auto r = R->data<T>();
    auto s = S->data<T>();
    // mutable_data分配内存、获取指针
    auto y = out_t->mutable_data<T>(ctx.GetPlace());

    const int total_num = V->dims()[0];
    const int k2 = R->dims()[1];

    for (int i = 0; i < V->numel(); ++i) {
      int fea_index = i % total_num;
      int sample_index = i / total_num;
      float sumsum = 0.0;
      for (int j = 0; j< k2; ++j) {
          int cur_fea_index = int(r[sample_index*k2+j]) * total_num + fea_index;
          sumsum += v[cur_fea_index] * s[sample_index*k2+j];
      }
      y[i] = sumsum;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
// 注册前向和反向op
REGISTER_OPERATOR(qe,
                  ops::QeOp,
                  ops::QeOpMaker,
                  paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
                  paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
// 注册CPU的Kernel
REGISTER_OP_CPU_KERNEL(qe,
                       ops::QeKernel<CPU, float>,
                       ops::QeKernel<CPU, double>);
