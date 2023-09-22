#ifndef SLAM_IN_AUTO_DRIVING_FLANN_HPP
#define SLAM_IN_AUTO_DRIVING_FLANN_HPP

#include "common/eigen_types.h"
#include "common/math_utils.h"
#include "common/point_types.h"
#include "nanoflann.hpp"

#include <glog/logging.h>
#include <execution>
#include <map>

//
// Created by sean on 2021/8/25.
//


namespace sad {

/**
 * 栅格法最近邻
 * @tparam dim 模板参数，使用2D或3D栅格
 */
template <typename T>
struct PointCloud_flann
{
    struct Point
    {
        T x, y, z;
    };

    using coord_t = T;  //!< The type of each coordinate

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

template <typename T>
class nanoflannNN {
   public:
    /// 设置点云，建立栅格
    bool SetPointCloud(CloudPtr cloud);

    /// 获取最近邻
    bool GetClosestPoint(const PointType& pt, PointType& closest_pt, size_t& idx, int knn_num_);

    /// 对比两个点云
    bool GetClosestPointForCloud(CloudPtr query, std::vector<std::pair<size_t, size_t>>& matches, int knn_num);

   private:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<T, PointCloud_flann<T>>,
        PointCloud_flann<T>, 3>;

    PointCloud_flann<T> cloud_flann;
    // CloudPtr cloud_;
    my_kd_tree_t index(3, cloud_flann, {10});
};

template <typename T>
bool nanoflannNN<T>::SetPointCloud(CloudPtr cloud) {
    cloud_flann.pts.resize(cloud->size());
    std::vector<size_t> num_index(cloud->size());
    std::for_each(num_index.begin(), num_index.end(), [idx = 0](size_t& i) mutable { i = idx++; });

    std::for_each(num_index.begin(), num_index.end(), [&cloud, this](const size_t& idx) {
        auto pt = cloud->points[idx];
        cloud_flann.pts[idx].x = pt.x;
        cloud_flann.pts[idx].y = pt.y;
        cloud_flann.pts[idx].z = pt.z;
    });
    // cloud_ = cloud;
    index = my_kd_tree_t(3, cloud_flann, {10});
    index.buildIndex();

    return true;
}

template <typename T>
bool nanoflannNN<T>::GetClosestPoint(const PointType& pt, PointType& closest_pt, size_t& idx, int knn_num_) {
    // do a knn search
    // std::cout << "GetClosestPoint" << std::endl;
    
    // const size_t                   num_results = knn_num_;
    // std::vector<size_t>            ret_index(num_results);
    // T                              out_dist_sqr;
    // nanoflann::KNNResultSet<T> resultSet(num_results);
    // resultSet.init(&ret_index[0], &out_dist_sqr);
    // index.findNeighbors(resultSet, query_pt);

    T query_pt[3] = {pt.x, pt.y, pt.z};
    size_t                num_results = 5;              // 最近的5个点
    std::vector<uint32_t> ret_index(num_results);       // 返回的点的索引
    std::vector<float>    out_dist_sqr(num_results);    // 返回的点的距离

    // 调用knnSearch()函数，返回最近的5个点的索引和距离
    num_results = index.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
    return true;
}

template <typename T>
bool nanoflannNN<T>::GetClosestPointForCloud(CloudPtr query,
                                          std::vector<std::pair<size_t, size_t>>& matches,
                                          int knn_num) {
    matches.clear();
    std::vector<size_t> index(query->size());
    std::for_each(index.begin(), index.end(), [idx = 0](size_t& i) mutable { i = idx++; });
    std::for_each(index.begin(), index.end(), [this, &matches, &query, knn_num](const size_t& idx) {
        PointType cp;
        size_t cp_idx;
        if (GetClosestPoint(query->points[idx], cp, cp_idx, knn_num)) {
            matches.emplace_back(cp_idx, idx);
        }
    });

    return true;
}

}

#endif  // SLAM_IN_AUTO_DRIVING_FLANN_HPP