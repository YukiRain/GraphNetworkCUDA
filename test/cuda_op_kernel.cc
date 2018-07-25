#include "tensorflow/core/framework/op.h"  
#include "tensorflow/core/framework/op_kernel.h"  

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("AddOne")
.Attr("T: {int32, float, double}")
.Input("input: T")
.Output("output: T")
.Attr("K: int")
.Doc(R"doc(  
Adds 1 to all elements of the tensor.  
output: A Tensor.  
  out_{i<K} = in_{i<K} + 1
)doc");

template<typename T>
void AddOneKernelLauncher(const T* in, const int batch_size, const int N, const int K, T* out);

template <typename Device, typename T>
class AddOneOp : public OpKernel {
public:
	explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("K", &K_));
	}

	void Compute(OpKernelContext* context) override {
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<T>();
		const int batch_size = input_tensor.shape().dim_size(0);
		const int N = input.size() / batch_size;
		
		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
			&output_tensor));
		auto output = output_tensor->flat<T>();
        OP_REQUIRES(context, K_>0 && K_<=N,
		            ::tensorflow::errors::InvalidArgument("Invalid K value"));
		AddOneKernelLauncher<T>(input.data(), batch_size, N, K_, output.data());
	}

private:
	int K_;
};

#ifndef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU).TypeConstraint<int>("T"),
                        AddOneOp<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        AddOneOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        AddOneOp<CPUDevice, double>);
#endif
