import numpy as np
import ctypes as ct
import h5py
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, 'graph_op.so'), '.')

def gen_graph(data, k=20):
    '''
    :param data: float32 np.ndarray which has shape (batch_size, num_points, 3)
    :param k: an integer representing the number of the nearest neighbor for constructing the graph
    :return: an float32 np.ndarray type adjacency matrix of the graph
    '''
    num_point = data.shape[0]
    c_num_point = ct.c_int(num_point)
    c_data = data.ctypes.data_as(ct.c_void_p)
    adjacency = np.zeros((num_point, num_point), dtype=np.float32)
    c_adjacency = adjacency.ctypes.data_as(ct.c_void_p)
    c_k = ct.c_int(k)
    dll.gen_graph(c_data, c_num_point, c_k, c_adjacency)
    return adjacency

def gen_graphs(data, k=20):
    '''
    :param data: float32 np.ndarray which has shape (batch_size, num_points, 3)
    :param k: an integer representing the number of the nearest neighbor for constructing the graph
    :return: numbers of float32 np.ndarray type adjacency matrices of the graph
    '''
    num_graphs = data.shape[0]
    num_point = data.shape[1]
    c_num_graphs = ct.c_int(num_graphs)
    c_num_point = ct.c_int(num_point)
    c_data = data.ctypes.data_as(ct.c_void_p)
    adjacency = np.zeros((num_graphs, num_point, num_point), dtype=np.float32)
    c_adjacency = adjacency.ctypes.data_as(ct.c_void_p)
    c_k = ct.c_int(k)
    dll.gen_graphs(c_data, c_num_point, c_num_graphs, c_k, c_adjacency)
    return adjacency

def gen_laplacians(data, k=30):
    '''
    :param data: float32 np.ndarray which has shape (batch_size, num_points, 3)
    :param k: an integer representing the number of the nearest neighbor for constructing the graph
    :return: an float32 np.ndarray type adjacency matrix of the graph
    '''
    num_graphs = data.shape[0]
    num_point = data.shape[1]
    c_num_graphs = ct.c_int(num_graphs)
    c_num_point = ct.c_int(num_point)
    c_data = data.ctypes.data_as(ct.c_void_p)
    laplacian = np.zeros((num_graphs, num_point, num_point), dtype=np.float32)
    c_laplacian = laplacian.ctypes.data_as(ct.c_void_p)
    c_k = ct.c_int(k)
    dll.gen_graphs(c_data, c_num_point, c_num_graphs, c_k, c_laplacian)
    return laplacian

def adj_test(data):
    ans = []
    for it in range(data.shape[0]):
        piece = np.mat(data[it, :, :])
        ans.append(np.array(piece*piece.T)[None, :, :])
    return np.concatenate(ans, 0)


if __name__ == '__main__':
    f = h5py.File('/home/zcx/Documents/datasets/ModelNet/ply_data_train2.h5', 'r')
    data_arr = f['data'].value[100: 110, :, :]
    start_1 = datetime.now()
    adj = gen_graphs(data_arr, 100)
    end_1 = datetime.now()
    # print(adj, np.sum(adj == 0), adj.shape)
    start_2 = datetime.now()
    adj2 = adj_test(data_arr)
    end_2 = datetime.now()
    start_3 = datetime.now()
    laplacian = gen_laplacians(data_arr, k=400)
    end_3 = datetime.now()
    time_1 = end_1 - start_1
    time_2 = end_2 - start_2
    time_3 = end_3 - start_3
    print(time_1, time_2, time_3)
    print('')
    print(laplacian, laplacian.shape)


