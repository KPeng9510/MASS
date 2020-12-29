// pybind libraries
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

// C/C++ includes
#include <cmath>
#include <cfloat>
#include <vector>
#include <chrono>
#include <math.h>
// #include <omp.h>

namespace py = pybind11;

namespace Eigen {
    typedef Matrix<bool, Dynamic, 1> VectorXb;
    typedef Matrix<signed char,Dynamic,1> VectorXsc;
};

Eigen::VectorXf _compute_dense_gt(const Eigen::MatrixXf & original_points,
                                     const Eigen::VectorXf & pc_range,
                                     const Eigen::VectorXf & voxel_size,
                                     const int class_number)
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
    const int vxsize = 1000; //(pxmax - pxmin) / voxel_size[0] +1;
    const int vysize = 500; //(pymax - pymin) / voxel_size[1] +1;
    const int vzsize = (pzmax - pzmin) / voxel_size[2];
    //const Eigen::Vector3i grid_size(vxsize, vysize, vzsize);
    //std::cout<<"1234_testtest"<<std::endl;
    //
    const int G1 = vxsize;
    const int G2 = vysize * G1; //result in y*x format
    //const int G3 = vzsize * GO2;
    const int G3 = 20 * G2;//result format class_number*y*x
    //
    //Eigen::RowVector3f offset3d(pxmin, pymin, pzmin);
    //Eigen::RowVector4f offset4d(pxmin, pymin, pzmin, 0.0f);

    //
    //std::cout<<G3<<std::endl;
    Eigen::VectorXf dense_gt = Eigen::VectorXf::Constant(G3, 0.0);
    //std::cout<<dense_gt.cols()<<std::endl;
    // go through all sampled points and put them into different bins based on the timestamps
    //std::vector<std::vector<int>> original_indices (T, std::vector<int>());
    for (int i=0; i< original_points.rows(); ++i) {
        
        const double pt_x = original_points(i,0) - pxmin;
        const double pt_y = original_points(i,1) - pymin;
        const double pt_z = original_points(i,2) - pzmin;
        if ((pt_x < 0)||(pt_x >= (pxmax-pxmin-0.1))||(pt_y < 0)||(pt_y >= (pymax-pymin-0.1))||(pt_z < 0) || (pt_z >= (pzmax-pzmin-0.1)))
        {
            continue;
         }
        //const double pt_z = original_points(i,2);
        const int pt_class = original_points(i,4);

        const int index_x = ceil(pt_x/voxel_size[0]);
        const int index_y = ceil(pt_y/voxel_size[1]);
        //const double index_z = pt_z/voxel_size[2];
        
        const int vector_index = (int)pt_class * G2 + index_y * G1 + index_x;
        //std::cout<<vector_index<<std::endl;
        if ((vector_index < 0)||(vector_index) > dense_gt.rows()){
            std::cout<<index_x<<std::endl;
            std::cout<<index_y<<std::endl;
            std::cout<<pt_class<<std::endl;
            std::cout<<vector_index<<std::endl;
            //goto k;
        }
        dense_gt[vector_index]++;
    }
    k:;
    return dense_gt;
}
PYBIND11_MODULE(dense, m) {
    m.doc() = "LiDAR voxelization";
    m.def("compute_dense_gt",
          &_compute_dense_gt,
          py::arg("original_points"),
          py::arg("pc_range"),
          py::arg("voxel_size"),
          py::arg("class_number")
          );
}
