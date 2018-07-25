import numpy as np
import ctypes as ct
import h5py
import os, sys
import tensorflow as tf
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dll = tf.load_op_library('./adj_gen.so')
print(dll)

f = h5py.File('/home/zcx/Documents/datasets/ModelNet/ply_data_train0.h5', 'r')
arr = f['data'].value[: 128, :, :]
a = tf.placeholder(tf.float32, [None, 2048, 3], name='a')
ans = dll.graph_adjacency_generator(a, 40)

print(ans)

with tf.Session() as sess:
    res = sess.run(ans, feed_dict={a: arr})
    print(res.shape, res.dtype)
