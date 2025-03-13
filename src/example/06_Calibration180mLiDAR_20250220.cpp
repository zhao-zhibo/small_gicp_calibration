// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT

/// @brief Basic point cloud registration example with small_gicp::align()
#include <queue>
#include <iostream>

#include <vector>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>

#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/factors/plane_icp_factor.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>

using PointT = pcl::PointXYZ;
using namespace small_gicp;

/// @brief Basic registration example using small_gicp::Registration.
void example1(const std::vector<Eigen::Vector4f>& target_points, const std::vector<Eigen::Vector4f>& source_points) {
  int num_threads = 4;                       // Number of threads to be used
  double downsampling_resolution = 0.25;     // Downsampling resolution
  int num_neighbors = 10;                    // Number of neighbor points used for normal and covariance estimation
  double max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

  // Convert to small_gicp::PointCloud
  auto target = std::make_shared<PointCloud>(target_points);
  auto source = std::make_shared<PointCloud>(source_points);

  // Downsampling
  target = voxelgrid_sampling_omp(*target, downsampling_resolution, num_threads);
  source = voxelgrid_sampling_omp(*source, downsampling_resolution, num_threads);

  // Create KdTree
  auto target_tree = std::make_shared<KdTree<PointCloud>>(target, KdTreeBuilderOMP(num_threads));
  auto source_tree = std::make_shared<KdTree<PointCloud>>(source, KdTreeBuilderOMP(num_threads));

  // Estimate point covariances
  estimate_covariances_omp(*target, *target_tree, num_neighbors, num_threads);
  estimate_covariances_omp(*source, *source_tree, num_neighbors, num_threads);

  // GICP + OMP-based parallel reduction
  Registration<GICPFactor, ParallelReductionOMP> registration;
  registration.reduction.num_threads = num_threads;
  registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;

  // Align point clouds
  Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
  auto result = registration.align(*target, *source, *target_tree, init_T_target_source);

  std::cout << "--- T_target_source ---" << std::endl << result.T_target_source.matrix() << std::endl;
  std::cout << "converged:" << result.converged << std::endl;
  std::cout << "error:" << result.error << std::endl;
  std::cout << "iterations:" << result.iterations << std::endl;
  std::cout << "num_inliers:" << result.num_inliers << std::endl;
  std::cout << "--- H ---" << std::endl << result.H << std::endl;
  std::cout << "--- b ---" << std::endl << result.b.transpose() << std::endl;
}

/** Custom registration example **/

/// @brief Point structure with mean, normal, and features.
struct MyPoint {
  std::array<double, 3> point;      // Point coorindates
  std::array<double, 3> normal;     // Point normal
  std::array<double, 36> features;  // Point features
};

/// @brief My point cloud class.
using MyPointCloud = std::vector<MyPoint>;

// Define traits for MyPointCloud so that it can be fed to small_gicp algorithms.
namespace small_gicp {
namespace traits {
template <>
struct Traits<MyPointCloud> {
  // *** Getters ***
  // The following getters are required for feeding this class to registration algorithms.

  // Number of points in the point cloud.
  static size_t size(const MyPointCloud& points) { return points.size(); }
  // Check if the point cloud has points.
  static bool has_points(const MyPointCloud& points) { return !points.empty(); }
  // Check if the point cloud has normals.
  static bool has_normals(const MyPointCloud& points) { return !points.empty(); }

  // Get i-th point. The last element should be 1.0.
  static Eigen::Vector4d point(const MyPointCloud& points, size_t i) {
    const auto& p = points[i].point;
    return Eigen::Vector4d(p[0], p[1], p[2], 1.0);
  }
  // Get i-th normal. The last element should be 0.0.
  static Eigen::Vector4d normal(const MyPointCloud& points, size_t i) {
    const auto& n = points[i].normal;
    return Eigen::Vector4d(n[0], n[1], n[2], 0.0);
  }
  // To use GICP, the following covariance getters are required additionally.
  // static bool has_covs(const MyPointCloud& points) { return !points.empty(); }
  // static const Eigen::Matrix4d cov(const MyPointCloud& points, size_t i);

  // *** Setters ***
  // The following methods are required for feeding this class to preprocessing algorithms.
  // (e.g., downsampling and normal estimation)

  // Resize the point cloud. This must retain the values of existing points.
  static void resize(MyPointCloud& points, size_t n) { points.resize(n); }
  // Set i-th point. pt = [x, y, z, 1.0].
  static void set_point(MyPointCloud& points, size_t i, const Eigen::Vector4d& pt) { Eigen::Map<Eigen::Vector3d>(points[i].point.data()) = pt.head<3>(); }
  // Set i-th normal. n = [nx, ny, nz, 0.0].
  static void set_normal(MyPointCloud& points, size_t i, const Eigen::Vector4d& n) { Eigen::Map<Eigen::Vector3d>(points[i].normal.data()) = n.head<3>(); }
  // To feed this class to estimate_covariances(), the following setter is required additionally.
  // static void set_cov(MyPointCloud& points, size_t i, const Eigen::Matrix4d& cov);
};
}  // namespace traits
}  // namespace small_gicp

/// @brief Custom nearest neighbor search with brute force search. (Do not use this in practical applications)
struct MyNearestNeighborSearch {
public:
  MyNearestNeighborSearch(const std::shared_ptr<MyPointCloud>& points) : points(points) {}

  size_t knn_search(const Eigen::Vector4d& pt, int k, size_t* k_indices, double* k_sq_dists) const {
    // Priority queue to hold top-k neighbors
    const auto comp = [](const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) { return lhs.second < rhs.second; };
    std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double>>, decltype(comp)> queue(comp);

    // Push pairs of (index, squared distance) to the queue
    for (size_t i = 0; i < points->size(); i++) {
      const double sq_dist = (Eigen::Map<const Eigen::Vector3d>(points->at(i).point.data()) - pt.head<3>()).squaredNorm();
      queue.push({i, sq_dist});
      if (queue.size() > k) {
        queue.pop();
      }
    }

    // Pop results
    const size_t n = queue.size();
    while (!queue.empty()) {
      k_indices[queue.size() - 1] = queue.top().first;
      k_sq_dists[queue.size() - 1] = queue.top().second;
      queue.pop();
    }

    return n;
  }

  std::shared_ptr<MyPointCloud> points;
};

// Define traits for MyNearestNeighborSearch so that it can be fed to small_gicp algorithms.
namespace small_gicp {
namespace traits {
template <>
struct Traits<MyNearestNeighborSearch> {
  /// @brief Find k-nearest neighbors.
  /// @note  This generic knn search is used for preprocessing (e.g., normal estimation).
  /// @param search      Nearest neighbor search
  /// @param point       Query point [x, y, z, 1.0]
  /// @param k           Number of neighbors
  /// @param k_indices   [out] Indices of the k-nearest neighbors
  /// @param k_sq_dists  [out] Squared distances of the k-nearest neighbors
  /// @return            Number of neighbors found
  static size_t knn_search(const MyNearestNeighborSearch& search, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return search.knn_search(point, k, k_indices, k_sq_dists);
  }

  /// @brief Find the nearest neighbor. This is a special case of knn_search with k=1 and is used in point cloud registration.
  ///        You can define this to optimize the search speed for k=1. Otherwise, nearest_neighbor_search() automatically falls back to knn_search() with k=1.
  ///        It is also valid to define only nearest_neighbor_search() and do not define knn_search() if you only feed your class to registration but not to preprocessing.
  /// @param search      Nearest neighbor search
  /// @param point       Query point [x, y, z, 1.0]
  /// @param k_indices   [out] Indices of the k-nearest neighbors
  /// @param k_sq_dists  [out] Squared distances of the k-nearest neighbors
  /// @return            1 if the nearest neighbor is found, 0 otherwise
  // static size_t nearest_neighbor_search(const MyNearestNeighborSearch& search, const Eigen::Vector4d& point, size_t* k_index, double* k_sq_dist);
};
}  // namespace traits
}  // namespace small_gicp

/// @brief Custom correspondence rejector.
struct MyCorrespondenceRejector {
  MyCorrespondenceRejector() : max_correpondence_dist_sq(1.0), min_feature_cos_dist(0.9) {}

  /// @brief Check if the correspondence should be rejected.
  /// @param T              Current estimate of T_target_source
  /// @param target_index   Target point index
  /// @param source_index   Source point index
  /// @param sq_dist        Squared distance between the points
  /// @return               True if the correspondence should be rejected
  bool operator()(const MyPointCloud& target, const MyPointCloud& source, const Eigen::Isometry3d& T, size_t target_index, size_t source_index, double sq_dist) const {
    // Reject correspondences with large distances
    if (sq_dist > max_correpondence_dist_sq) {
      return true;
    }

    // You can define your own rejection criteria here (e.g., based on features)
    Eigen::Map<const Eigen::Matrix<double, 36, 1>> target_features(target[target_index].features.data());
    Eigen::Map<const Eigen::Matrix<double, 36, 1>> source_features(target[target_index].features.data());
    if (target_features.dot(source_features) < min_feature_cos_dist) {
      return true;
    }

    return false;
  }

  double max_correpondence_dist_sq;  // Maximum correspondence distance
  double min_feature_cos_dist;       // Maximum feature distance
};

/// @brief Custom general factor that can control the registration process.
struct MyGeneralFactor {
  MyGeneralFactor() : lambda(1e8) {}

  /// @brief Update linearized system.
  /// @note  This method is  called just before the linearized system is solved.
  ///        By modifying the linearized system (H, b, e), you can inject arbitrary constraints.
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param target_tree  Nearest neighbor search for the target point cloud
  /// @param T            Linearization point
  /// @param H            [in/out] Linearized information matrix.
  /// @param b            [in/out] Linearized information vector.
  /// @param e            [in/out] Error at the linearization point.
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree>
  void update_linearized_system(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry3d& T,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* e) const {
    // Optimization DoF mask [rx, ry, rz, tx, ty, tz] (1.0 = inactive, 0.0 = active)
    Eigen::Matrix<double, 6, 1> dof_mask;
    dof_mask << 1.0, 1.0, 0.0, 0.0, 0.0, 0.0;

    // Fix roll and pitch rotation by adding a large penalty (soft contraint)
    (*H) += dof_mask.asDiagonal() * lambda;
  }

  /// @brief Update error consisting of per-point factors.
  /// @note  This method is  called just after the linearized system is solved in LM to check if the objective function is decreased.
  ///        If you modified the error in update_linearized_system(), you need to update the error here for consistency.
  /// @param target   Target point cloud
  /// @param source   Source point cloud
  /// @param T        Evaluation point
  /// @param e        [in/out] Error at the evaluation point.
  template <typename TargetPointCloud, typename SourcePointCloud>
  void update_error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, double* e) const {
    // No update is required for the error.
  }

  double lambda;  ///< Regularization parameter (Increasing this makes the constraint stronger)
};

// 将 pcl::PointCloud<pcl::PointXYZ>::Ptr 转换为 std::vector<Eigen::Vector4f>
std::vector<Eigen::Vector4f> convertToEigenPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::vector<Eigen::Vector4f> eigen_points;
    for (const auto& point : cloud->points) {
        eigen_points.emplace_back(point.x, point.y, point.z, 1.0f);
    }
    return eigen_points;
}

// 将 std::vector<Eigen::Vector4f> 转换回 pcl::PointCloud<pcl::PointXYZ>::Ptr
pcl::PointCloud<pcl::PointXYZ>::Ptr convertToPCLPoints(const std::vector<Eigen::Vector4f>& eigen_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& point : eigen_points) {
        cloud->points.emplace_back(point.x(), point.y(), point.z());
    }
    return cloud;
}

// 将 MyPointCloud 转换为 pcl::PointCloud<pcl::PointXYZ>::Ptr
pcl::PointCloud<pcl::PointXYZ>::Ptr convertMyPointCloudToPCL(const MyPointCloud& myPointCloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& myPoint : myPointCloud) {
        pcl::PointXYZ pclPoint;
        pclPoint.x = myPoint.point[0];
        pclPoint.y = myPoint.point[1];
        pclPoint.z = myPoint.point[2];
        pclCloud->points.push_back(pclPoint);
    }
    return pclCloud;
}

// 将点云中的所有点乘以变换矩阵
pcl::PointCloud<pcl::PointXYZ>::Ptr transformPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudPoints, const Eigen::Matrix4d& transformMatrix) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedPoints(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& point : cloudPoints->points) {
        Eigen::Vector4f homogenousPoint(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f transformedPoint = transformMatrix.cast<float>() * homogenousPoint;
        pcl::PointXYZ pclPoint;
        pclPoint.x = transformedPoint.x();
        pclPoint.y = transformedPoint.y();
        pclPoint.z = transformedPoint.z();
        transformedPoints->points.push_back(pclPoint);
    }
    return transformedPoints;
}

void visualizeAlignment(const pcl::PointCloud<PointT>::Ptr& sourcePoints,
                        const pcl::PointCloud<PointT>::Ptr& targetPoints,
                        const pcl::PointCloud<PointT>::Ptr& alignedPoints) {
    pcl::visualization::PCLVisualizer viewer("ICP Alignment");
    int v1(0), v2(1);
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

    float bckgrColor = 0.0;
    float txtColor = 1.0 - bckgrColor;

    // 目标点云（绿色）
    pcl::visualization::PointCloudColorHandlerCustom<PointT> targetColor(targetPoints, 0, 255, 0);
    viewer.addPointCloud(targetPoints, targetColor, "target_v1", v1);
    viewer.addPointCloud(targetPoints, targetColor, "target_v2", v2);

    // 原始点云（蓝色）
    pcl::visualization::PointCloudColorHandlerCustom<PointT> sourceColor(sourcePoints, 0, 0, 255);
    viewer.addPointCloud(sourcePoints, sourceColor, "source_v1", v1);

    // ICP 对齐后的点云（红色）
    pcl::visualization::PointCloudColorHandlerCustom<PointT> alignedColor(alignedPoints, 255, 0, 0);
    viewer.addPointCloud(alignedPoints, alignedColor, "aligned_v2", v2);

    // 添加文本描述
    viewer.addText("Green: Target cloud\nBlue: Source cloud", 10, 15, 16, txtColor, txtColor, txtColor, "info_v1", v1);
    viewer.addText("Green: Target cloud\nRed: Aligned cloud", 10, 15, 16, txtColor, txtColor, txtColor, "info_v2", v2);

    viewer.setBackgroundColor(bckgrColor, bckgrColor, bckgrColor, v1);
    viewer.setBackgroundColor(bckgrColor, bckgrColor, bckgrColor, v2);
    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(960, 540);

    viewer.spin();
}

/// @brief Example to perform preprocessing and registration separately.
void CalibrateLiDAR(const std::vector<Eigen::Vector4f>& target_points, const std::vector<Eigen::Vector4f>& source_points) {
  int num_threads = 32;                       // Number of threads to be used
  double downsampling_resolution = 0.5;     // Downsampling resolution
  int num_neighbors = 10;                    // Number of neighbor points used for normal and covariance estimation
  double max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

  std::cout << "Start CalibrateLiDAR "<< std::endl;
  // Convert to MyPointCloud
  std::shared_ptr<MyPointCloud> target = std::make_shared<MyPointCloud>();
  target->resize(target_points.size());
  for (size_t i = 0; i < target_points.size(); i++) {
    Eigen::Map<Eigen::Vector3d>(target->at(i).point.data()) = target_points[i].head<3>().cast<double>();
  }

  std::shared_ptr<MyPointCloud> source = std::make_shared<MyPointCloud>();
  source->resize(source_points.size());
  for (size_t i = 0; i < source_points.size(); i++) {
    Eigen::Map<Eigen::Vector3d>(source->at(i).point.data()) = source_points[i].head<3>().cast<double>();
  }

  // Downsampling works directly on MyPointCloud
  target = voxelgrid_sampling_omp(*target, downsampling_resolution , num_threads);
  source = voxelgrid_sampling_omp(*source, downsampling_resolution , num_threads);

  // Create nearest neighbor search
  auto target_search = std::make_shared<MyNearestNeighborSearch>(target);
  auto source_search = std::make_shared<MyNearestNeighborSearch>(target);

  // Estimate point normals
  // You can use your custom nearest neighbor search here!
  std::cout << "Start Estimate Target point normals"<< std::endl;
  estimate_normals_omp(*target, *target_search, num_neighbors, num_threads);
  std::cout << "Start Estimate Source point normals"<< std::endl;
  estimate_normals_omp(*source, *source_search, num_neighbors, num_threads);

  // Compute point features (e.g., FPFH, SHOT, etc.)
  for (size_t i = 0; i < target->size(); i++) {
    target->at(i).features.fill(1.0);
  }
  for (size_t i = 0; i < source->size(); i++) {
    source->at(i).features.fill(1.0);
  }

  // Point-to-plane ICP + OMP-based parallel reduction
  using PerPointFactor = PointToPlaneICPFactor;             // Use point-to-plane ICP factor. Target must have normals.
  using Reduction = ParallelReductionOMP;                   // Use OMP-based parallel reduction
  using GeneralFactor = MyGeneralFactor;                    // Use custom general factor
  using CorrespondenceRejector = MyCorrespondenceRejector;  // Use custom correspondence rejector
  using Optimizer = GaussNewtonOptimizer;            // Use Levenberg-Marquardt optimizer

  Registration<PerPointFactor, Reduction, GeneralFactor, CorrespondenceRejector, Optimizer> registration;
  registration.reduction.num_threads = num_threads;
  registration.rejector.max_correpondence_dist_sq = max_correspondence_distance * max_correspondence_distance;
  registration.general_factor.lambda = 1e8;

  // Align point clouds
  // Again, you can use your custom nearest neighbor search here!
  Eigen::Matrix3d R;
  R << -5.7232600e-01,  8.1904800e-01, -4.0054000e-02,
  -8.0827600e-01, -5.7169000e-01, -1.4092600e-01,
  -1.3832300e-01, -4.8281000e-02,  9.8921000e-01;
  Eigen::Vector3d T;
  T << -2.5151617e+01, 7.9413929e+01,  3.7523700e+00;
  // 检查旋转矩阵是否正交
  if (!R.isUnitary()) {
      std::cerr << "Error: Rotation matrix R is not unitary." << std::endl;
  }
  // 检查旋转矩阵的行列式是否为1（即是否为有效的旋转矩阵）
  if (std::abs(R.determinant() - 1.0) > 1e-6) {
      std::cerr << "Error: Rotation matrix R determinant is not 1." << std::endl;
  }
  std::cout << "Start align "<< std::endl;
  Eigen::Isometry3d init_T_target_source;
  init_T_target_source.linear() = R;
  init_T_target_source.translation() = T;
  auto result = registration.align(*target, *source, *target_search, init_T_target_source);

  std::cout << "--- T_target_source ---" << std::endl << result.T_target_source.matrix() << std::endl;
  std::cout << "converged:" << result.converged << std::endl;
  std::cout << "error:" << result.error << std::endl;
  std::cout << "iterations:" << result.iterations << std::endl;
  std::cout << "num_inliers:" << result.num_inliers << std::endl;
  std::cout << "--- H ---" << std::endl << result.H << std::endl;
  std::cout << "--- b ---" << std::endl << result.b.transpose() << std::endl;

  // 显示对齐前和对齐后的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr sourcePoints = convertMyPointCloudToPCL(*source);
  pcl::PointCloud<pcl::PointXYZ>::Ptr targetPoints = convertMyPointCloudToPCL(*target);
  pcl::PointCloud<pcl::PointXYZ>::Ptr alignedPoints = transformPointCloud(sourcePoints,result.T_target_source.matrix());
  visualizeAlignment(sourcePoints, targetPoints, alignedPoints);
}

int main(int argc, char** argv) {

  // 1. 读取 PCD 文件中的点云（LiDAR点云）
  pcl::PointCloud<pcl::PointXYZ>::Ptr LiDARPoints(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::io::loadPCDFile("/media/zhao/ZhaoZhibo1T/AllData/tunnelRoadside/Xu_180m/Calibration/1740039933.738297462.pcd", *LiDARPoints); // 点云的pcd路径
  std::cout << "Loaded LiDAR PCD point cloud with " << LiDARPoints->width * LiDARPoints->height << " data points." << std::endl;
  
  // 2. 读取 LAS 文件中的点云（地图点云）
  pcl::PointCloud<pcl::PointXYZ>::Ptr MapPoints(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::io::loadPCDFile("/media/zhao/ZhaoZhibo1T/AllData/tunnelRoadside/HDMap/HDMap_LiDAR_90m_And_180m/180mLidar1.pcd", *MapPoints); // 高精地图的pcd路径
  // 对 MapPoints 点云中的每个点进行处理
  for (auto& point : MapPoints->points) {
      point.x -= 513500;
      point.y -= 3364400;
  }
  std::cout << "Loaded LAS map point cloud with " << MapPoints->width * MapPoints->height << " data points." << std::endl;

  // 将 LiDARPoints 转换为 source_points
  std::vector<Eigen::Vector4f> source_points = convertToEigenPoints(LiDARPoints);
  // 将 MapPoints 转换为 target_points
  std::vector<Eigen::Vector4f> target_points = convertToEigenPoints(MapPoints);

  if (target_points.empty() || source_points.empty()) {
      std::cerr << "error: failed to read points from data/(target|source).ply" << std::endl;
      return 0;
  }
  // example1(target_points, source_points);
  CalibrateLiDAR(target_points, source_points);

  return 0;
}