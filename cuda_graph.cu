#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <queue>
#include <vector>
#include <functional>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

typedef unsigned int uint32;
typedef unsigned long long int uint64;
const int __INT_MAX = 0x7fffffff;

////////////////////////////////////////////////////////////////////////////////
// This class is defined for the convenience of priority_queue
// The attribute x and y are for saving the indices of two nodes,
// and the attribute dist is for saving and comparing the distance between the nodes.
template <class T> class Node_t {
public:
	T dist;
	uint64 x, y;

	Node_t() {}
	Node_t(T _d, uint64 _x, uint64 _y) : dist(_d), x(_x), y(_y) {}
	__device__ bool operator<(const Node_t<T>& n) const { return dist < n.dist; }
	__device__ bool operator>(const Node_t<T>& n) const { return dist > n.dist; }
	__device__ bool operator<=(const Node_t<T>& n) const { return dist <= n.dist; }
	__device__ bool operator>=(const Node_t<T>& n) const { return dist >= n.dist; }
};

/////////////////////////////////////////////////////////////////////////////
// Here we begin our implementation of the graph adjacency matrix generation.
template <typename T>
__device__ __inline__ T distance(const T* a, const T* b) {
	return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]);
}

// The CUDA kernel of the implementation
template <typename T>
__global__ void GraphGenerator(const T* in, int* out, int k_val, uint64 num_points, uint64 channels) {
	// Since the input has shape [batch_size, num_points, channels], 
	// we use batch_size numbers of blocks, and num_points numbers of threads.
	uint64 in_begin = blockIdx.x * blockDim.x * channels;
	uint64 out_begin = blockIdx.x * blockDim.x * k_val + threadIdx.x * k_val;
	T* heap = new T[k_val];	

	// Initialization of the output tensor
	for(int i=0; i<k_val; i++) {
		out[out_begin + i] = __INT_MAX;
	}

	// The time complexity is not optimal, yet k_val is often a small integer,
	// this solution may also be acceptable.
	for(int i=0; i<num_points; i++) {
		if(i == threadIdx.x)
			continue;
		T tmp = distance(&in[in_begin + threadIdx.x*channels], &in[in_begin + i*channels]);
		for(int j=0; j<k_val; j++) {
			if(tmp < heap[j]) {
				heap[j] = tmp;
				out[out_begin + j] = i;
			}
		}
	}
	delete heap;
}

template <typename T>
void GraphLauncher(const T* input, int* output, int k_val, uint64 batch_size, uint64 num_points, uint64 channels) {
	GraphGenerator<<< batch_size, num_points >>>(input, output, k_val, num_points, channels);
	cudaDeviceSynchronize();
}


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

		// GraphLaucher(input.data(), output.data(), k_val, num_points, channels);
		GraphLauncher<T>(input.data(), output.data(), K_, batch_size, num_points, channels);
	}

private:
	int K_;

};

REGISTER_KERNEL_BUILDER(Name("GraphAdjacencyGenerator").Device(DEVICE_GPU).TypeConstraint<float>("T"), GraphAdjacencyGeneratorOp<GPUDevice, float>);
//REGISTER_KERNEL_BUILDER(Name("GraphAdjacencyGenerator").Device(DEVICE_GPU).TypeConstraint<double>("T"), GraphAdjacencyGeneratorOp<GPUDevice, double>);

#endif
