// pybind libraries
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// C/C++ includes
#include <cmath>
#include <cfloat>
#include <vector>
#include <chrono>

// #include <omp.h>

namespace py = pybind11;

namespace Eigen {
    typedef Matrix<bool, Dynamic, 1> VectorXb;
    typedef Matrix<signed char,Dynamic,1> VectorXsc;
};

Eigen::VectorXf _compute_dense_gt(const Eigen::MatrixXf & original_points,
                                     const Eigen::VectorXf & pc_range,
                                     const Eigen::VectorXf & voxel_size,
                                     const double class_number)
{
    // py::gil_scoped_acquire acquire;

    //
    const double & pxmin = pc_range[0];
    const double & pymin = pc_range[1];
    const double & pzmin = pc_range[2];
    const double & pxmax = pc_range[3];
    const double & pymax = pc_range[4];
    const double & pzmax = pc_range[5];

    //
    const int vxsize = (pxmax - pxmin) / voxel_size[0];
    const int vysize = (pymax - pymin) / voxel_size[1];
    const int vzsize = (pzmax - pzmin) / voxel_size[2];
    const Eigen::Vector3i grid_size(vxsize, vysize, vzsize);

    //
    const int G1 = vxsize;
    const int G2 = vysize * G1; //result in y*x format
    //const int G3 = vzsize * G2;
    const int G3 = class_number * G2;//result format class_number*y*x
    //
    Eigen::RowVector3f offset3d(pxmin, pymin, pzmin);
    Eigen::RowVector4f offset4d(pxmin, pymin, pzmin, 0.0f);

    //
    Eigen::VectorXf dense_gt = Eigen::VectorXf::Constant(G3, 0.0);

    // go through all sampled points and put them into different bins based on the timestamps
    //std::vector<std::vector<int>> original_indices (T, std::vector<int>());
    for (int i=0; i< original_points.rows(); ++i) {

        const double pt_x = original_points(i,0);
        const double pt_y = original_points(i,1);
        //const double pt_z = original_points(i,2);
        const double pt_class = original_points(i,3);
        //const double index_x = pt_x/voxel_size[0];
        //const double index_y = pt_y/voxel_size[1];
        //const double index_z = pt_z/voxel_size[2];

        const double vector_index = pt_class * G2 + pt_y * G1 + pt_x;
        dense_gt[vector_index]++;
    }

    return dense_gt;
}
PYBIND11_MODULE(mapping, m) {
    m.doc() = "LiDAR voxelization";
    m.def("compute_dense_gt",
          &_compute_dense_gt,
          py::arg("original_points"),
          py::arg("pc_range"),
          py::arg("voxel_size"),
          py::arg("class_number")
          );
}
