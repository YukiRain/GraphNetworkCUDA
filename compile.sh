TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()[: -1]))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_IFLAGS=( $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')  )

ABI_FLAG=-D_GLIBCXX_USE_CXX11_ABI=0

g++ -std=c++11 -shared tf_graph.cc -o tf_graph.so -fPIC ${TF_CFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_LFLAGS[@]} -O2

nvcc -std=c++11 -c cuda_graph.cu -o cuda_graph.cu.o -c -g -D GOOGLE_CUDA=1 ${TF_CFLAGS[@]}\
 -x cu -Xcompiler -fPIC -I/usr/local/cuda-9.1/include -I$TF_IFLAGS -I$TF_IFLAGS/external/nsync/public $ABI_FLAG

g++ -std=c++11 -shared -o adj_gen.so tf_graph.cc cuda_graph.cu.o -D_GLIBCXX_USE_CXX11_ABI=0\
 -I$TF_IFLAGS -I$TF_IFLAGS/external/nsync/public -L/usr/local/cuda-9.1/lib64 ${TF_LFLAGS[@]} -fPIC -lcudart

python test.py

rm tf_graph.so
rm cuda_graph.cu.o
