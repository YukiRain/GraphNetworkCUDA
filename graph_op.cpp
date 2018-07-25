#include <iostream>
#include <cmath>
#include <algorithm>
#include <queue>
#include <vector>
#include <string>
#include <functional>

using namespace std;

typedef unsigned int uint32;

class Node {
public:
	float dist;
	uint32 x, y;

	Node(float _d, uint32 _x, uint32 _y) :dist(_d), x(_x), y(_y) {}
	bool operator<(const Node& n) const { return dist < n.dist; }
	bool operator>(const Node& n) const { return dist > n.dist; }
	bool operator<=(const Node& n) const { return dist <= n.dist; }
	bool operator>=(const Node& n) const { return dist >= n.dist; }
};

class pointData {
	float* data;
	uint32 num, col = 0;

	typedef struct {
		string info = "Index out of range!";
	}OutOfRangeError;

	typedef struct {
		string info = "Input parameters must be one of 'x' or 'y' or 'z'";
	}InvalidParameterError;

public:
	pointData() :data(NULL), num(0) {}
	pointData(float* _input, uint32 _num) :data(_input), num(_num) {}
	uint32& column() { return this->col; }

	float distance(uint32 i, uint32 j) {
		pointData& item = *this;
		float *a = item(i), *b = item(j);
		float ans = (a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]) + (a[2] - b[2])*(a[2] - b[2]);
		return sqrt(ans);
	}

	float* end() const {
		if (col == 0)
			return data + 3 * num;
		else
			return data + num;
	}

	void set_data(float* _data) { data = _data; }

	// for vector with 3*num floats
	float* operator()(uint32 idx) const {
		if (idx >= num) {
			OutOfRangeError e;
			throw e;
		}
		return data + 3 * idx;
	}

	// for vector with 3*num floats
	float& operator()(uint32 idx, char c) {
		if (idx >= num) {
			OutOfRangeError e;
			throw e;
		}
		pointData& item = *this;
		if (c == 'x')
			return *(data + 3 * idx);
		else if (c == 'y')
			return *(data + 3 * idx + 1);
		else if (c == 'z')
			return *(data + 3 * idx + 2);
		else {
			InvalidParameterError e;
			throw e;
		}
	}

	// for col*col matrix
	float& operator()(uint32 i, uint32 j, uint32 col) {
		if (i*col + j >= num) {
			OutOfRangeError e;
			throw e;
		}
		return *(data + i*col + j);
	}

	friend ostream& operator<<(ostream& os, pointData& pdata);
};

ostream& operator<<(ostream& os, pointData& pdata) {
	os << "---------ADJACENCY MATRIX----------" << endl;
	for (uint32 i = 0; i < pdata.col; i++) {
		for (uint32 j = 0; j < pdata.col; j++) {
			os << pdata(i, j, pdata.col) << "\t";
		}
		os << endl;
	}
	os << "-----------------------------------" << endl;
	return os;
}

extern "C" {
/**
* @brief	Generate a graph based on k-nearest neighbor search
* @param	points	three dimensional points representing the point cloud
* @param	num		the number of points
* @param	k		the number of nearest neighbors to compute
* @param	output	a num*num matrix representing the adjacency matrix for the graph
**/
void gen_graph(float* points, int num, int k, float* output) {
	pointData data(points, num), out(output, num*num);
	out.column() = num;
	priority_queue<Node, vector<Node>, greater<Node>> que;
	for (int i = 0; i < num; i++) {
		for (int j = 0; j < num; j++) {
			if (i == j) {
				out(i, j, num) = 0;
				continue;
			}
			float tmp = data.distance(i, j);
			que.push(Node(tmp, i, j));
			if (que.size() > k)
				que.pop();
		}
		while (!que.empty()) {
			out(que.top().x, que.top().y, num) = que.top().dist;
			out(que.top().y, que.top().x, num) = que.top().dist;
			que.pop();
		}
	}
}

/**
* @brief	Generate graphs based on k-nearest neighbor search
* @param	points			three dimensional points representing the point cloud
* @param	num_points		the number of points
* @param	num_graphs		the number of graphs
* @param	k				the number of nearest neighbors to compute
* @param	output			num_graphs number of num*num matrices representing the adjacency matrix for the graph
**/
void gen_graphs(float* points, int num_points, int num_graphs, uint32 k, float* output) {
	pointData data(points, num_points), out(output, num_points * num_points);
	vector<pointData> data_vec, out_vec;
	out.column() = num_points;
	data_vec.push_back(data);
	out_vec.push_back(out);
	for (uint32 idx = 1; idx < num_graphs; idx++) {
		data_vec.push_back(pointData(data_vec.back().end(), num_points));
		out_vec.push_back(pointData(out_vec.back().end(), num_points * num_points));
		out_vec.back().column() = num_points;
	}
	for (uint32 idx = 0; idx < num_graphs; idx++) {
		priority_queue<Node, vector<Node>, greater<Node>> que;
		for (int i = 0; i < num_points; i++) {
			for (int j = 0; j < num_points; j++) {
				if (i == j) {
					out_vec[idx](i, j, num_points) = 0;
					continue;
				}
				float tmp = data_vec[idx].distance(i, j);
				que.push(Node(tmp, i, j));
				if (que.size() > k)
					que.pop();
			}
			while (!que.empty()) {
				out_vec[idx](que.top().x, que.top().y, num_points) = que.top().dist;
				out_vec[idx](que.top().y, que.top().x, num_points) = que.top().dist;
				que.pop();
			}
		}
	}
}

/**
* @brief	Generate graphs based on k-nearest neighbor search
* @param	points			three dimensional points representing the point cloud
* @param	num_points		the number of points
* @param	num_graphs		the number of graphs
* @param	k				the number of nearest neighbors to compute
* @param	output			num_graphs of num*num Laplacian matrices for the graph
**/
void gen_laplacian(float* points, int num_points, int num_graphs, uint32 k, float* output) {
	pointData data(points, num_points), out(output, num_points * num_points);
	vector<pointData> data_vec, out_vec;
	out.column() = num_points;
	data_vec.push_back(data);
	out_vec.push_back(out);
	for (uint32 idx = 1; idx < num_graphs; idx++) {
		data_vec.push_back(pointData(data_vec.back().end(), num_points));
		out_vec.push_back(pointData(out_vec.back().end(), num_points * num_points));
		out_vec.back().column() = num_points;
	}
	for (uint32 idx = 0; idx < num_graphs; idx++) {
		priority_queue<Node, vector<Node>, greater<Node>> que;
		for (int i = 0; i < num_points; i++) {
			for (int j = 0; j < num_points; j++) {
				if (i == j) {
					out_vec[idx](i, j, num_points) = 0;
					continue;
				}
				float tmp = data_vec[idx].distance(i, j);
				que.push(Node(tmp, i, j));
				if (que.size() > k)
					que.pop();
			}
			out_vec[idx](i, i, num_points) = float(que.size());
			while (!que.empty()) {
				out_vec[idx](que.top().x, que.top().y, num_points) = -que.top().dist;
				out_vec[idx](que.top().y, que.top().x, num_points) = -que.top().dist;
				que.pop();
			}
		}
	}
}

} // extern "C"

//int _test1() {
//	int m, n;
//	while (cin >> m >> n) {
//		float* pdata = new float[m*n * 3];
//		float* pout = new float[m*n*n];
//		for (int i = 0; i < m; i++) {
//			for (int j = 0; j < n; j++) {
//				cin >> pdata[3 * (i*n + j)] >> pdata[3 * (i*n + j) + 1] >> pdata[3 * (i*n + j) + 2];
//			}
//		}
//		gen_graphs(pdata, n, m, 2, pout);
//		pointData view(pout, n*n);
//		view.column() = n;
//		for (int i = 0; i < m; i++) {
//			cout << view << endl;
//			view.set_data(view.end());
//		}
//		delete pdata;
//		delete pout;
//	}
//	return 0;
//}
//
//int _test2() {
//	int n;
//	while (cin >> n) {
//		float* pdata = new float[n * 3];
//		float* pout = new float[n*n];
//		for (int j = 0; j < n; j++) {
//			cin >> pdata[3 * j] >> pdata[3 * j + 1] >> pdata[3 * j + 2];
//		}
//		gen_graph(pdata, n, 2, pout);
//		pointData view(pout, n*n);
//		view.column() = n;
//		cout << view << endl;
//		delete pdata;
//		delete pout;
//	}
//	return 0;
//}
//
//int test3() {
//	int m, n;
//	while (cin >> m >> n) {
//		float* pdata = new float[m*n * 3];
//		float* pout = new float[m*n*n];
//		float* plaplacian = new float[m*n*n];
//		for (int i = 0; i < m; i++) {
//			for (int j = 0; j < n; j++) {
//				cin >> pdata[3 * (i*n + j)] >> pdata[3 * (i*n + j) + 1] >> pdata[3 * (i*n + j) + 2];
//			}
//		}
//		gen_graphs(pdata, n, m, 2, pout);
//		gen_laplacian(pdata, n, m, 2, plaplacian);
//		pointData view_adj(pout, n*n), view_laplacian(plaplacian, n*n);
//		view_adj.column() = n;
//		view_laplacian.column() = n;
//		for (int i = 0; i < m; i++) {
//			cout << view_adj << endl;
//			view_adj.set_data(view_adj.end());
//		}
//		cout << "=====================================" << endl << endl;
//		for (int i = 0; i < m; i++) {
//			cout << view_laplacian << endl;
//			view_laplacian.set_data(view_laplacian.end());
//		}
//		delete pdata;
//		delete pout;
//		delete plaplacian;
//	}
//	return 0;
//}
//
//int main() {
//	test3();
//	return 0;
//}