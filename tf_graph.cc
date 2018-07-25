#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <iostream>
#include <functional>
#include <string>
#include <algorithm>
#include <queue>

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

typedef unsigned int uint32;
typedef unsigned long long int uint64;

REGISTER_OP("GraphAdjacencyGenerator")
.Attr("T: {float, double}")
.Input("input: T")
.Attr("K: int")
.Output("output: int32")
.Doc(R"doc(
	Input a set of 3D points, and output its adjacency matrix.
)doc");


template <typename T>
void GraphLauncher(const T* input, int* output, int k_val,
	uint64 batch_size, uint64 point_num, uint64 channels);

////////////////////////////////////////////////////////////////////////////////////
// The tensorflow OpKernel implementation wrapper.
template <typename Device, typename T>
class GraphAdjacencyGeneratorOp : public OpKernel {
public:
	explicit GraphAdjacencyGeneratorOp(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("K", &K_));
		//OP_REQUIRES(context, (K_>0 && K_<=num_points), 
		//	::tensorflow::errors::InvalidArgument("The value of K must be an integer in range [1, num_points]"));
	}

	void Compute(OpKernelContext* context) override {
		const Tensor& input_tensor = context->input(0);
		OP_REQUIRES(context, input_tensor.dims() == 3,
			::tensorflow::errors::InvalidArgument("GraphAdjacencyGenerator expects (batch_size, num_points, 3) points shape"));
		uint64 batch_size = input_tensor.shape().dim_size(0);
		uint64 num_points = input_tensor.shape().dim_size(1);
		uint64 channels = input_tensor.shape().dim_size(2);
		OP_REQUIRES(context, channels == 3,
			::tensorflow::errors::InvalidArgument("The last dimension for the input op must be 3"));
		auto input = input_tensor.flat<T>();

		// Validate the value of K_
		//OP_REQUIRES_OK(context, context->GetAttr("K"), &K_);
		OP_REQUIRES(context, (K_>0 && K_<=num_points), 
			::tensorflow::errors::InvalidArgument("The value of K must be an integer in range [1, num_points]"));

		Tensor* output_tensor = NULL;
		::tensorflow::TensorShape output_shape;
		output_shape.AddDim(batch_size);
		output_shape.AddDim(num_points);
		output_shape.AddDim(K_);
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
		auto output = output_tensor->template flat<int>();

		GraphLauncher(input.data(), output.data(), K_, batch_size, num_points, channels);
		// GraphGenerator<<< batch_size, num_points >>>(input.data(), output.data(), K_, num_points, channels);
	}

private:
	int K_;

};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T) \
	REGISTER_KERNEL_BUILDER(Name("GraphAdjacencyGenerator").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
	GraphAdjacencyGenerator<T>);

//REGISTER_GPU(float);
//REGISTER_GPU(double);

REGISTER_KERNEL_BUILDER(Name("GraphAdjacencyGenerator").Device(DEVICE_GPU).TypeConstraint<float>("T"), GraphAdjacencyGeneratorOp<GPUDevice, float>);
//REGISTER_KERNEL_BUILDER(Name("GraphAdjacencyGenerator").Device(DEVICE_GPU).TypeConstraint<double>("T"), GraphAdjacencyGeneratorOp<GPUDevice, double>);

#else
REGISTER_KERNEL_BUILDER(Name("GraphAdjacencyGenerator").Device(DEVICE_CPU).TypeConstraint<float>("T"), GraphAdjacencyGeneratorOp<CPUDevice, float>);

#endif
