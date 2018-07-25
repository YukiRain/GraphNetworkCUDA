TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()[: -1]))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_IFLAGS=( $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')  )

g++ -std=c++11 -shared cuda_op_kernel.cc -o cuda_op_kernel.so -fPIC ${TF_CFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0\
 ${TF_LFLAGS[@]} -O2
# g++ -std=c++11 -shared cuda_op_kernel.cc -o cuda_op_kernel.so -fPIC -I ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc ${TF_CFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0\
 ${TF_LFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc cuda_op_kernel.cu.o -D_GLIBCXX_USE_CXX11_ABI=0\
 -I$TF_IFLAGS -I$TF_IFLAGS/external/nsync/public -L/usr/local/cuda-9.1/lib64 ${TF_LFLAGS[@]} -fPIC -lcudart


python test.py
