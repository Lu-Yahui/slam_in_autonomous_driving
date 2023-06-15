#include "scan_context.h"

namespace scan_context {
namespace {

template <typename T>
inline T RadToDeg(T radians) {
    return radians * 180.0 / M_PI;
}

template <typename T>
inline T DegToRad(T degrees) {
    return degrees * M_PI / 180.0;
}

template <typename T>
inline T XyToTheta(T x, T y) {
    if (x >= 0.0 && y >= 0.0) {
        return (180.0 / M_PI) * std::atan(y / x);
    }

    if (x < 0.0 && y >= 0.0) {
        return 180.0 - ((180.0 / M_PI) * std::atan(y / (-x)));
    }

    if (x < 0.0 && y < 0.0) {
        return 180.0 + ((180.0 / M_PI) * std::atan(y / x));
    }

    if (x >= 0.0 & y < 0.0) {
        return 360.0 - ((180.0 / M_PI) * std::atan((-y) / x));
    }
}

Eigen::MatrixXd CircShift(const Eigen::MatrixXd &mat, uint32_t num_shift) {
    if (num_shift == 0U) {
        return mat;
    }

    Eigen::MatrixXd shifted_mat = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());
    for (int c = 0; c < mat.cols(); ++c) {
        int new_location = (c + static_cast<int>(num_shift)) % mat.cols();
        shifted_mat.col(new_location) = mat.col(c);
    }
    return shifted_mat;
}

std::vector<float> EigenMatToStdVector(const Eigen::MatrixXd &eigmat) {
    std::vector<float> vec(eigmat.data(), eigmat.data() + eigmat.size());
    return vec;
}

}  // namespace

ScanContextManager::ScanContextManager(const Options &options) : options_(options) {}

void ScanContextManager::MakeAndSaveScancontextAndKeys(const PointCloud<SCPointType> &scan) {
    Eigen::MatrixXd sc = MakeScanContext(scan);
    Eigen::MatrixXd ring_key = MakeRingKeyFromScanContext(sc);
    Eigen::MatrixXd sector_key = MakeSectorKeyFromScanContext(sc);
    std::vector<float> polar_context_invkey_vec = EigenMatToStdVector(ring_key);

    polar_contexts_.push_back(sc);
    polar_context_invkeys_.push_back(ring_key);
    polar_context_vkeys_.push_back(sector_key);
    polar_context_invkeys_mat_.push_back(polar_context_invkey_vec);
}

double ScanContextManager::DistDirectSC(const Eigen::MatrixXd &sc1, const Eigen::MatrixXd &sc2) {
    int num_eff_cols = 0;  // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for (int col_idx = 0; col_idx < sc1.cols(); col_idx++) {
        Eigen::VectorXd col_sc1 = sc1.col(col_idx);
        Eigen::VectorXd col_sc2 = sc2.col(col_idx);

        if (col_sc1.norm() == 0 | col_sc2.norm() == 0) {
            continue;  // don't count this sector pair.
        }

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }

    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;
}

int ScanContextManager::FastAlignUsingVkey(const Eigen::MatrixXd &vkey1, const Eigen::MatrixXd &vkey2) {
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = 10000000;
    for (int shift_idx = 0; shift_idx < vkey1.cols(); shift_idx++) {
        Eigen::MatrixXd vkey2_shifted = CircShift(vkey2, shift_idx);
        Eigen::MatrixXd vkey_diff = vkey1 - vkey2_shifted;
        double cur_diff_norm = vkey_diff.norm();
        if (cur_diff_norm < min_veky_diff_norm) {
            argmin_vkey_shift = shift_idx;
            min_veky_diff_norm = cur_diff_norm;
        }
    }

    return argmin_vkey_shift;
}

std::pair<double, int> ScanContextManager::DistanceBetweenScanContext(const Eigen::MatrixXd &sc1,
                                                                      const Eigen::MatrixXd &sc2) {
    // 1. fast align using variant key (not in original IROS18)
    Eigen::MatrixXd vkey_sc1 = MakeSectorKeyFromScanContext(sc1);
    Eigen::MatrixXd vkey_sc2 = MakeSectorKeyFromScanContext(sc2);
    int argmin_vkey_shift = FastAlignUsingVkey(vkey_sc1, vkey_sc2);

    const int search_radius = std::round(0.5 * options_.search_ratio * sc1.cols());  // a half of search range
    std::vector<int> shift_idx_search_space{argmin_vkey_shift};
    for (int ii = 1; ii < search_radius + 1; ii++) {
        shift_idx_search_space.push_back((argmin_vkey_shift + ii + sc1.cols()) % sc1.cols());
        shift_idx_search_space.push_back((argmin_vkey_shift - ii + sc1.cols()) % sc1.cols());
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

    // 2. fast columnwise diff
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for (int num_shift : shift_idx_search_space) {
        Eigen::MatrixXd sc2_shifted = CircShift(sc2, num_shift);
        double cur_sc_dist = DistDirectSC(sc1, sc2_shifted);
        if (cur_sc_dist < min_sc_dist) {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }

    return {min_sc_dist, argmin_shift};
}

Eigen::MatrixXd ScanContextManager::MakeScanContext(const PointCloud<SCPointType> &scan) {
    constexpr double kNoPoint{-1000.0};
    Eigen::MatrixXd desc = kNoPoint * Eigen::MatrixXd::Ones(options_.num_of_rings, options_.num_of_sectors);
    for (uint32_t i = 0U; i < scan.points.size(); ++i) {
        const double x = scan.points[i].x;
        const double y = scan.points[i].y;
        const double z = scan.points[i].z + options_.lidar_height_m;

        const double azim_range = std::hypot(x, y);
        if (azim_range > options_.max_radius_m) {
            continue;
        }

        const double azim_angle = XyToTheta(x, y);

        const int ring_index = std::max(
            std::min(static_cast<int>(options_.num_of_rings),
                     static_cast<int>(std::ceil((azim_range / options_.max_radius_m) * options_.num_of_rings))),
            1);
        const int sector_index =
            std::max(std::min(static_cast<int>(options_.num_of_sectors),
                              static_cast<int>(std::ceil((azim_angle / 360.0) * options_.num_of_sectors))),
                     1);

        desc(ring_index - 1, sector_index - 1) = std::max(desc(ring_index - 1, sector_index - 1), z);
    }

    // reset no points to zero (for cosine dist later)
    for (int r = 0; r < desc.rows(); ++r) {
        for (int c = 0; c < desc.cols(); ++c) {
            if (desc(r, c) == kNoPoint) {
                desc(r, c) = 0.0;
            }
        }
    }
    return desc;
}

Eigen::MatrixXd ScanContextManager::MakeRingKeyFromScanContext(const Eigen::MatrixXd &desc) {
    // rowwise mean vector
    Eigen::MatrixXd invariant_key(desc.rows(), 1);
    for (int r = 0; r < desc.rows(); ++r) {
        invariant_key(r, 0) = desc.row(r).mean();
    }
    return invariant_key;
}

Eigen::MatrixXd ScanContextManager::MakeSectorKeyFromScanContext(const Eigen::MatrixXd &desc) {
    // columnwise mean vector
    Eigen::MatrixXd variant_key(1, desc.cols());
    for (int c = 0; c < desc.cols(); ++c) {
        variant_key(0, c) = desc.col(c).mean();
    }
    return variant_key;
}

const std::vector<Eigen::MatrixXd> &ScanContextManager::GetPolarContexts() const { return polar_contexts_; }

std::pair<int, double> ScanContextManager::DetectLoopClosureID() {
    auto curr_key = polar_context_invkeys_mat_.back();
    auto curr_desc = polar_contexts_.back();

    /*
     * step 1: candidates from ringkey tree_
     */
    if (polar_context_invkeys_mat_.size() < options_.num_of_exclude_recent + 1U) {
        return {-1, 0.0};
    }

    // tree_ reconstruction (not mandatory to make everytime)
    if (tree_making_period_conter_ % options_.tree_making_period == 0U) {
        polar_context_invkeys_to_search_.clear();
        polar_context_invkeys_to_search_.assign(polar_context_invkeys_mat_.begin(),
                                                polar_context_invkeys_mat_.end() - options_.num_of_exclude_recent);

        polar_context_tree_.reset();
        polar_context_tree_ =
            std::make_unique<InvKeyTree>(options_.num_of_rings, polar_context_invkeys_to_search_, 10 /* max leaf */);
    }
    tree_making_period_conter_ += 1U;

    double min_dist = 10000000.0;
    int nn_align = 0;
    int nn_idx = 0;

    // knn search
    std::vector<size_t> candidate_indices(options_.num_of_candidates_from_tree);
    std::vector<float> out_dists_sqr(options_.num_of_candidates_from_tree);

    nanoflann::KNNResultSet<float> knnsearch_result(options_.num_of_candidates_from_tree);
    knnsearch_result.init(&candidate_indices[0], &out_dists_sqr[0]);
    polar_context_tree_->index->findNeighbors(knnsearch_result, &curr_key[0], nanoflann::SearchParams(10));

    // pairwise distance (find optimal columnwise best-fit using cosine distance)
    for (const auto &candidate_index : candidate_indices) {
        const auto &polar_context_candidate = polar_contexts_[candidate_index];
        std::pair<double, int> sc_dist_result = DistanceBetweenScanContext(curr_desc, polar_context_candidate);

        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if (candidate_dist < min_dist) {
            min_dist = candidate_dist;
            nn_align = candidate_align;
            nn_idx = candidate_index;
        }
    }

    int loop_closure_id{-1};
    if (min_dist > options_.scan_context_dist_thresh) {
        return {-1, 0.0};
    }

    const double yaw_diff_rad = DegToRad(static_cast<double>(nn_align) * options_.angle_per_sector_deg);
    return {nn_idx, yaw_diff_rad};
}

}  // namespace scan_context