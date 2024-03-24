#pragma once

#include <atomic>
#include <memory>

#include <tbb/tbb.h>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

namespace small_gicp {

/// @brief Voxel grid downsampling using TBB.
/// @param points     Input points
/// @param leaf_size  Downsampling resolution
/// @return           Downsampled points
template <typename InputPointCloud, typename OutputPointCloud = InputPointCloud>
std::shared_ptr<OutputPointCloud> voxelgrid_sampling_tbb(const InputPointCloud& points, double leaf_size) {
  if (traits::size(points) == 0) {
    std::cerr << "warning: empty input points!!" << std::endl;
    return std::make_shared<OutputPointCloud>();
  }

  const double inv_leaf_size = 1.0 / leaf_size;
  const int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3 = 63bits in 64bit int)
  const size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
  const int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive

  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(points.size());
  tbb::parallel_for(size_t(0), size_t(traits::size(points)), [&](size_t i) {
    // TODO: Check if coord in 21bit range
    const Eigen::Vector4d pt = traits::point(points, i);
    const Eigen::Array4i coord = (pt * inv_leaf_size).array().floor().template cast<int>() + coord_offset;

    // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
    const std::uint64_t bits =                                 //
      ((coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |  //
      ((coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |  //
      ((coord[2] & coord_bit_mask) << (coord_bit_size * 2));
    coord_pt[i] = {bits, i};
  });

  // Sort by voxel coords
  tbb::parallel_sort(coord_pt, [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  auto downsampled = std::make_shared<OutputPointCloud>();
  traits::resize(*downsampled, traits::size(points));

  // Take block-wise sum
  const int block_size = 1024;
  std::atomic_uint64_t num_points = 0;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, traits::size(points), block_size), [&](const tbb::blocked_range<size_t>& range) {
    std::vector<Eigen::Vector4d> sub_points;
    sub_points.reserve(block_size);

    Eigen::Vector4d sum_pt = traits::point(points, coord_pt[range.begin()].second);
    for (size_t i = range.begin() + 1; i != range.end(); i++) {
      if (coord_pt[i - 1].first != coord_pt[i].first) {
        sub_points.emplace_back(sum_pt / sum_pt.w());
        sum_pt.setZero();
      }
      sum_pt += traits::point(points, coord_pt[i].second);
    }
    sub_points.emplace_back(sum_pt / sum_pt.w());

    const size_t point_index_begin = num_points.fetch_add(sub_points.size());
    for (size_t i = 0; i < sub_points.size(); i++) {
      traits::set_point(*downsampled, point_index_begin + i, sub_points[i]);
    }
  });

  traits::resize(*downsampled, num_points);

  return downsampled;
}

}  // namespace small_gicp
