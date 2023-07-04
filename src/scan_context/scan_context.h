#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "KDTreeVectorOfVectorsAdaptor.h"
#include "nanoflann.hpp"

namespace scan_context {

struct PointXYZI {
    double x;
    double y;
    double z;
    double i;
};

template <typename T>
struct PointCloud {
    std::vector<T> points;
};

// using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for
// detail, refer 20 ICRA Intensity Scan Context)
using SCPointType = PointXYZI;
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

class ScanContextManager {
   public:
    struct Options {
        // add this for simply directly using lidar scan in the lidar local coord (not robot base coord), if you use
        // robot-coord-transformed lidar scans, just set this as 0.
        double lidar_height_m = 0.0;
        // 20 in the original paper (IROS 18)
        uint32_t num_of_rings = 20U;
        // 60 in the original paper (IROS 18)
        uint32_t num_of_sectors = 60U;
        // 80 meter max in the original paper (IROS 18)
        double max_radius_m = 80.0;
        double angle_per_sector_deg = 360.0 / static_cast<double>(num_of_sectors);
        double ring_gap_m = max_radius_m / static_cast<double>(num_of_rings);
        // simply just keyframe gap, but node position distance-based exclusion is ok.
        uint32_t num_of_exclude_recent = 1U;
        // 10 is enough. (refer the IROS 18 paper)
        uint32_t num_of_candidates_from_tree = 10U;
        // for fast comparison, no Brute-force, but search 10% is okay. Not in the original conf paper, but improved
        // version.
        double search_ratio = 0.1;
        // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score
        // check (e.g., in LeGO-LOAM) should be required for robustness)
        double scan_context_dist_thresh = 0.13;
        // remaking tree frequency, to avoid non-mandatory every remaking, to save time cost. Ff you want to find a very
        // recent revisits use small value of it (it is enough fast ~ 5-50ms wrt N.).
        uint32_t tree_making_period = 50U;
    };

   public:
    ScanContextManager(const Options &options);

    void MakeAndSaveScancontextAndKeys(const PointCloud<SCPointType> &scan);

    /**
     * @brief
     *
     * @return std::pair<int, double> nearest node index, relative yaw
     */
    std::pair<int, double> DetectLoopClosureID();

    const std::vector<Eigen::MatrixXd> &GetPolarContexts() const;

   private:
    Eigen::MatrixXd MakeScanContext(const PointCloud<SCPointType> &scan);

    Eigen::MatrixXd MakeRingKeyFromScanContext(const Eigen::MatrixXd &desc);

    Eigen::MatrixXd MakeSectorKeyFromScanContext(const Eigen::MatrixXd &desc);

    int FastAlignUsingVkey(const Eigen::MatrixXd &vkey1, const Eigen::MatrixXd &vkey2);

    /**
     * @brief "d" (eq 5) in the original paper (IROS 18)
     *
     * @param sc1
     * @param sc2
     * @return double
     */
    double DistDirectSC(const Eigen::MatrixXd &sc1, const Eigen::MatrixXd &sc2);

    /**
     * @brief "D" (eq 6) in the original paper (IROS 18)
     *
     * @param sc1
     * @param sc2
     * @return std::pair<double, int>
     */
    std::pair<double, int> DistanceBetweenScanContext(const Eigen::MatrixXd &sc1, const Eigen::MatrixXd &sc2);

   private:
    Options options_;
    uint32_t tree_making_period_conter_ = 0U;

    // data
    std::vector<double> polar_contexts_timestamp_;
    std::vector<Eigen::MatrixXd> polar_contexts_;
    std::vector<Eigen::MatrixXd> polar_context_invkeys_;
    std::vector<Eigen::MatrixXd> polar_context_vkeys_;

    KeyMat polar_context_invkeys_mat_;
    KeyMat polar_context_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polar_context_tree_;
};

}  // namespace scan_context
