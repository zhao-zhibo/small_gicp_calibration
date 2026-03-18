// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT

/// @brief 使用 small_gicp::align() 的点云配准示例
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

/// @brief 使用 small_gicp::Registration 的基础配准示例
void example1(const std::vector<Eigen::Vector4f>& target_points, const std::vector<Eigen::Vector4f>& source_points) {
  int num_threads = 4;                        // 使用的线程数
  double downsampling_resolution = 0.1;      // 体素下采样分辨率
  int num_neighbors = 20;                    // 法向量和协方差估计使用的邻居点数量
  double max_correspondence_distance = 3.0;  // 最大对应点距离

  // 转换为 small_gicp::PointCloud
  auto target = std::make_shared<PointCloud>(target_points);
  auto source = std::make_shared<PointCloud>(source_points);

  // 体素下采样
  target = voxelgrid_sampling_omp(*target, downsampling_resolution, num_threads);
  source = voxelgrid_sampling_omp(*source, downsampling_resolution, num_threads);

  // 创建 KdTree
  auto target_tree = std::make_shared<KdTree<PointCloud>>(target, KdTreeBuilderOMP(num_threads));
  auto source_tree = std::make_shared<KdTree<PointCloud>>(source, KdTreeBuilderOMP(num_threads));

  // 估计点协方差
  estimate_covariances_omp(*target, *target_tree, num_neighbors, num_threads);
  estimate_covariances_omp(*source, *source_tree, num_neighbors, num_threads);

  // 使用 GICP + OMP 并行归约
  Registration<GICPFactor, ParallelReductionOMP> registration;
  registration.reduction.num_threads = num_threads;
  registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;

  // 执行配准
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

/** 自定义配准示例 **/

/// @brief 带有均值、法向量和特征的点结构
struct MyPoint {
  std::array<double, 3> point;      // 点坐标
  std::array<double, 3> normal;     // 点法向量
  std::array<double, 36> features;  // 点特征
};

/// @brief 自定义点云类型
using MyPointCloud = std::vector<MyPoint>;

// 为 MyPointCloud 定义 traits，使其可以被 small_gicp 算法直接使用
namespace small_gicp {
namespace traits {
template <>
struct Traits<MyPointCloud> {
  // 获取点云大小
  static size_t size(const MyPointCloud& points) { return points.size(); }
  // 判断是否有点
  static bool has_points(const MyPointCloud& points) { return !points.empty(); }
  // 判断是否有法向量
  static bool has_normals(const MyPointCloud& points) { return !points.empty(); }

  // 获取第 i 个点，最后一个分量固定为 1.0
  static Eigen::Vector4d point(const MyPointCloud& points, size_t i) {
    const auto& p = points[i].point;
    return Eigen::Vector4d(p[0], p[1], p[2], 1.0);
  }

  // 获取第 i 个法向量，最后一个分量固定为 0.0
  static Eigen::Vector4d normal(const MyPointCloud& points, size_t i) {
    const auto& n = points[i].normal;
    return Eigen::Vector4d(n[0], n[1], n[2], 0.0);
  }

  // 调整点云大小，并保留已有数据
  static void resize(MyPointCloud& points, size_t n) { points.resize(n); }
  // 设置第 i 个点
  static void set_point(MyPointCloud& points, size_t i, const Eigen::Vector4d& pt) { Eigen::Map<Eigen::Vector3d>(points[i].point.data()) = pt.head<3>(); }
  // 设置第 i 个法向量
  static void set_normal(MyPointCloud& points, size_t i, const Eigen::Vector4d& n) { Eigen::Map<Eigen::Vector3d>(points[i].normal.data()) = n.head<3>(); }
};
}  // namespace traits
}  // namespace small_gicp

/// @brief 使用暴力搜索实现的自定义近邻搜索（仅示例，不建议实际项目直接使用）
struct MyNearestNeighborSearch {
public:
  MyNearestNeighborSearch(const std::shared_ptr<MyPointCloud>& points) : points(points) {}

  size_t knn_search(const Eigen::Vector4d& pt, int k, size_t* k_indices, double* k_sq_dists) const {
    // 优先队列中保存当前最优的 top-k 邻居
    const auto comp = [](const std::pair<size_t, double>& lhs, const std::pair<size_t, double>& rhs) { return lhs.second < rhs.second; };
    std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double>>, decltype(comp)> queue(comp);

    // 将所有点与查询点的距离加入队列
    for (size_t i = 0; i < points->size(); i++) {
      const double sq_dist = (Eigen::Map<const Eigen::Vector3d>(points->at(i).point.data()) - pt.head<3>()).squaredNorm();
      queue.push({i, sq_dist});
      if (queue.size() > k) {
        queue.pop();
      }
    }

    // 输出搜索结果
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

// 为自定义近邻搜索定义 traits，使其可以被 small_gicp 算法调用
namespace small_gicp {
namespace traits {
template <>
struct Traits<MyNearestNeighborSearch> {
  /// @brief 查找 k 个最近邻
  static size_t knn_search(const MyNearestNeighborSearch& search, const Eigen::Vector4d& point, size_t k, size_t* k_indices, double* k_sq_dists) {
    return search.knn_search(point, k, k_indices, k_sq_dists);
  }
};
}  // namespace traits
}  // namespace small_gicp

/// @brief 自定义对应关系剔除器
struct MyCorrespondenceRejector {
  MyCorrespondenceRejector() : max_correpondence_dist_sq(1.0), min_feature_cos_dist(0.9) {}

  /// @brief 判断该对应关系是否需要被剔除
  bool operator()(const MyPointCloud& target, const MyPointCloud& source, const Eigen::Isometry3d& T, size_t target_index, size_t source_index, double sq_dist) const {
    // 距离过大则剔除
    if (sq_dist > max_correpondence_dist_sq) {
      return true;
    }

    // 这里可以扩展为更多自定义特征筛选规则
    Eigen::Map<const Eigen::Matrix<double, 36, 1>> target_features(target[target_index].features.data());
    Eigen::Map<const Eigen::Matrix<double, 36, 1>> source_features(target[target_index].features.data());
    if (target_features.dot(source_features) < min_feature_cos_dist) {
      return true;
    }

    return false;
  }

  double max_correpondence_dist_sq;  // 最大对应点距离平方
  double min_feature_cos_dist;       // 最小特征余弦相似度
};

/// @brief 可控制配准过程的自定义通用因子
struct MyGeneralFactor {
  MyGeneralFactor() : lambda(1e8) {}

  /// @brief 在线性化系统求解前，对 H 和 b 做额外约束注入
  template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree>
  void update_linearized_system(
    const TargetPointCloud& target,
    const SourcePointCloud& source,
    const TargetTree& target_tree,
    const Eigen::Isometry3d& T,
    Eigen::Matrix<double, 6, 6>* H,
    Eigen::Matrix<double, 6, 1>* b,
    double* e) const {
    // 优化自由度掩码 [rx, ry, rz, tx, ty, tz]，1 表示施加强约束
    Eigen::Matrix<double, 6, 1> dof_mask;
    dof_mask << 1.0, 1.0, 0.0, 0.0, 0.0, 0.0;

    // 通过大惩罚项约束 roll 和 pitch
    (*H) += dof_mask.asDiagonal() * lambda;
  }

  /// @brief 更新误差项；这里保持不变即可
  template <typename TargetPointCloud, typename SourcePointCloud>
  void update_error(const TargetPointCloud& target, const SourcePointCloud& source, const Eigen::Isometry3d& T, double* e) const {
    // 这里不需要额外更新误差
  }

  double lambda;  ///< 正则化参数，越大约束越强
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

// 使用给定变换矩阵对点云中所有点进行变换
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

// 可视化配准前后的点云结果
void visualizeAlignment(const pcl::PointCloud<PointT>::Ptr& sourcePoints,
                        const pcl::PointCloud<PointT>::Ptr& targetPoints,
                        const pcl::PointCloud<PointT>::Ptr& alignedPoints) {
    pcl::visualization::PCLVisualizer viewer("ICP Alignment");
    int v1(0), v2(1);
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

    float bckgrColor = 1.0;  // 背景色
    float txtColor = 1.0 - bckgrColor;

    // 目标点云使用绿色显示
    pcl::visualization::PointCloudColorHandlerCustom<PointT> targetColor(targetPoints, 0, 255, 0);
    viewer.addPointCloud(targetPoints, targetColor, "target_v1", v1);
    viewer.addPointCloud(targetPoints, targetColor, "target_v2", v2);

    // 原始车载点云使用蓝色显示
    pcl::visualization::PointCloudColorHandlerCustom<PointT> sourceColor(sourcePoints, 0, 0, 255);
    viewer.addPointCloud(sourcePoints, sourceColor, "source_v1", v1);

    // 配准后的车载点云使用红色显示
    pcl::visualization::PointCloudColorHandlerCustom<PointT> alignedColor(alignedPoints, 255, 0, 0);
    viewer.addPointCloud(alignedPoints, alignedColor, "aligned_v2", v2);

    // 设置点大小
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "target_v1", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "target_v2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "source_v1", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "aligned_v2", v2);

    // 添加文字说明
    viewer.addText("Green: RoadSide point cloud\nBlue: Vehicle point cloud", 10, 15, 16, txtColor, txtColor, txtColor, "info_v1", v1);
    viewer.addText("Green: RoadSide point cloud\nRed: Aligned Vehicle cloud", 10, 15, 16, txtColor, txtColor, txtColor, "info_v2", v2);

    viewer.setBackgroundColor(bckgrColor, bckgrColor, bckgrColor, v1);
    viewer.setBackgroundColor(bckgrColor, bckgrColor, bckgrColor, v2);
    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(960, 540);

    viewer.spin();
}

/// @brief 分别执行预处理和配准的主流程
void CalibrateLiDAR(const std::vector<Eigen::Vector4f>& target_points, const std::vector<Eigen::Vector4f>& source_points) {
  int num_threads = 8;                        // 使用的线程数
  double downsampling_resolution = 0.4;      // 体素下采样分辨率
  int num_neighbors = 10;                    // 法向量估计使用的邻居点数量
  double max_correspondence_distance = 1.0;  // 最大对应点距离

  std::cout << "Start CalibrateLiDAR " << std::endl;

  // 转换为自定义点云结构
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

  // 对目标点云进行下采样
  target = voxelgrid_sampling_omp(*target, downsampling_resolution, num_threads);
  // 如果需要，也可以对 source 做同样的下采样
  // source = voxelgrid_sampling_omp(*source, downsampling_resolution / 4, num_threads);

  // 在目标点云下采样后，输出当前参与配准的路侧点云和车载点云数量
  std::cout << "RoadSide point count after downsampling: " << target->size() << std::endl;
  std::cout << "Vehicle point count for registration: " << source->size() << std::endl;

  // 创建近邻搜索器
  auto target_search = std::make_shared<MyNearestNeighborSearch>(target);
  auto source_search = std::make_shared<MyNearestNeighborSearch>(source);

  // 估计法向量
  std::cout << "Start Estimate Target point normals" << std::endl;
  estimate_normals_omp(*target, *target_search, num_neighbors, num_threads);
  std::cout << "Start Estimate Source point normals" << std::endl;
  estimate_normals_omp(*source, *source_search, num_neighbors, num_threads);

  // 这里将所有特征先统一置为常数，仅用于保持与原示例一致
  for (size_t i = 0; i < target->size(); i++) {
    target->at(i).features.fill(1.0);
  }
  for (size_t i = 0; i < source->size(); i++) {
    source->at(i).features.fill(1.0);
  }

  // 点到平面 ICP + OMP 并行归约
  using PerPointFactor = PointToPlaneICPFactor;             // 使用点到平面 ICP 因子
  using Reduction = ParallelReductionOMP;                   // 使用 OMP 并行归约
  using GeneralFactor = MyGeneralFactor;                    // 使用自定义通用因子
  using CorrespondenceRejector = MyCorrespondenceRejector;  // 使用自定义对应剔除器
  using Optimizer = GaussNewtonOptimizer;                   // 使用高斯牛顿优化器

  Registration<PerPointFactor, Reduction, GeneralFactor, CorrespondenceRejector, Optimizer> registration;
  registration.reduction.num_threads = num_threads;
  registration.rejector.max_correpondence_dist_sq = max_correspondence_distance * max_correspondence_distance;
  registration.general_factor.lambda = 1e8;

  // 初始化的路侧雷达到车载雷达变换矩阵
  Eigen::Matrix4d T_roadLidar_Vehiclelidar;
  T_roadLidar_Vehiclelidar <<
      0.202303,  -0.272429,   0.940668,   -7.15308,
     -0.812974,  -0.582267,   0.00620889, -14.9781,
      0.546028,  -0.765995,  -0.339271,   -5.86747,
      0.0,        0.0,        0.0,        1.0;

  std::cout << "T_roadLidar_Vehiclelidar:" << std::endl << T_roadLidar_Vehiclelidar << std::endl;

  // 从初始矩阵中拆出旋转和平移
  Eigen::Matrix3d R = T_roadLidar_Vehiclelidar.block<3, 3>(0, 0);
  Eigen::Vector3d T = T_roadLidar_Vehiclelidar.block<3, 1>(0, 3);

  // 检查旋转矩阵是否正交
  if (!R.isUnitary()) {
      std::cerr << "Error: Rotation matrix R is not unitary." << std::endl;
  }

  // 检查旋转矩阵行列式是否为 1
  if (std::abs(R.determinant() - 1.0) > 1e-6) {
      std::cerr << "Error: Rotation matrix R determinant is not 1." << std::endl;
  }

  std::cout << "Start align " << std::endl;
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

  // 可视化配准前后的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr sourcePoints = convertMyPointCloudToPCL(*source);
  pcl::PointCloud<pcl::PointXYZ>::Ptr targetPoints = convertMyPointCloudToPCL(*target);
  pcl::PointCloud<pcl::PointXYZ>::Ptr alignedPoints = transformPointCloud(sourcePoints, result.T_target_source.matrix());
  visualizeAlignment(sourcePoints, targetPoints, alignedPoints);
}

int main(int argc, char** argv) {
  // 1. target：路侧点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr roadLiDARPoint(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::io::loadPCDFile("/home/zhao/Data/HK_data/Temp_large_compare/1751960570.2311_roadSide.pcd", *roadLiDARPoint);

  // 2. source：车载点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr vehileLiDARPoint(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::io::loadPCDFile("/home/zhao/Data/HK_data/Temp_large_compare/1751960570.2894_VehicleleCloud.pcd", *vehileLiDARPoint);

  // source 点云
  std::vector<Eigen::Vector4f> source_points = convertToEigenPoints(vehileLiDARPoint);
  // target 点云
  std::vector<Eigen::Vector4f> target_points = convertToEigenPoints(roadLiDARPoint);

  if (target_points.empty() || source_points.empty()) {
      std::cerr << "error: failed to read points from input pcd files" << std::endl;
      return 0;
  }

  // example1(target_points, source_points);
  CalibrateLiDAR(target_points, source_points);

  return 0;
}
