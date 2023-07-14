//
// Created by xiang on 2022/7/14.
//

#include "ndt_3d.h"
#include "common/math_utils.h"

#include <glog/logging.h>
#include <Eigen/SVD>
#include <execution>
#include <fstream>

namespace sad {

void Ndt3d::BuildVoxels() {
    assert(target_ != nullptr);
    assert(target_->empty() == false);
    grids_.clear();

    /// 分配体素
    std::vector<size_t> index(target_->size());
    std::for_each(index.begin(), index.end(), [idx = 0](size_t& i) mutable { i = idx++; });

    std::for_each(index.begin(), index.end(), [this](const size_t& idx) {
        auto pt = ToVec3d(target_->points[idx]);
        auto key = (pt * options_.inv_voxel_size_).cast<int>();
        if (grids_.find(key) == grids_.end()) {
            grids_.insert({key, {idx}});
        } else {
            grids_[key].idx_.emplace_back(idx);
        }
    });

    /// 计算每个体素中的均值和协方差
    std::for_each(std::execution::par_unseq, grids_.begin(), grids_.end(), [this](auto& v) {
        if (v.second.idx_.size() > options_.min_pts_in_voxel_) {
            // 要求至少有３个点
            math::ComputeMeanAndCov(v.second.idx_, v.second.mu_, v.second.sigma_,
                                    [this](const size_t& idx) { return ToVec3d(target_->points[idx]); });
            // SVD 检查最大与最小奇异值，限制最小奇异值

            Eigen::JacobiSVD svd(v.second.sigma_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vec3d lambda = svd.singularValues();
            if (lambda[1] < lambda[0] * 1e-3) {
                lambda[1] = lambda[0] * 1e-3;
            }

            if (lambda[2] < lambda[0] * 1e-3) {
                lambda[2] = lambda[0] * 1e-3;
            }

            Mat3d inv_lambda = Vec3d(1.0 / lambda[0], 1.0 / lambda[1], 1.0 / lambda[2]).asDiagonal();

            // v.second.info_ = (v.second.sigma_ + Mat3d::Identity() * 1e-3).inverse();  // 避免出nan
            v.second.info_ = svd.matrixV() * inv_lambda * svd.matrixU().transpose();
        }
    });

    /// 删除点数不够的
    for (auto iter = grids_.begin(); iter != grids_.end();) {
        if (iter->second.idx_.size() > options_.min_pts_in_voxel_) {
            iter++;
        } else {
            iter = grids_.erase(iter);
        }
    }
}

bool Ndt3d::AlignNdt(SE3& init_pose) {
    LOG(INFO) << "aligning with ndt";
    assert(grids_.empty() == false);

    SE3 pose = init_pose;
    if (options_.remove_centroid_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
        LOG(INFO) << "init trans set to " << pose.translation().transpose();
    }

    // 对点的索引，预先生成
    int num_residual_per_point = 1;
    if (options_.nearby_type_ == NearbyType::NEARBY6) {
        num_residual_per_point = 7;
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    // 我们来写一些并发代码
    int total_size = index.size() * num_residual_per_point;

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        std::vector<bool> effect_pts(total_size, false);
        std::vector<Eigen::Matrix<double, 3, 6>> jacobians(total_size);
        std::vector<Vec3d> errors(total_size);
        std::vector<Mat3d> infos(total_size);
        std::vector<double> scores(total_size, 0.0);

        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Vec3d qs = pose * q;  // 转换之后的q

            // 计算qs所在的栅格以及它的最近邻栅格
            Vec3i key = (qs * options_.inv_voxel_size_).cast<int>();
            for (int i = 0; i < nearby_grids_.size(); ++i) {
                auto key_off = key + nearby_grids_[i];
                auto it = grids_.find(key_off);
                int real_idx = idx * num_residual_per_point + i;
                if (it != grids_.end()) {
                    auto& v = it->second;  // voxel
                    Vec3d e = qs - v.mu_;

                    // check chi2 th
                    double res = e.transpose() * v.info_ * e;
                    if (std::isnan(res) || res > options_.res_outlier_th_) {
                        effect_pts[real_idx] = false;
                        continue;
                    }

                    // build residual
                    Eigen::Matrix<double, 3, 6> J;
                    J.block<3, 3>(0, 0) = -pose.so3().matrix() * SO3::hat(q);
                    J.block<3, 3>(0, 3) = Mat3d::Identity();

                    jacobians[real_idx] = J;
                    errors[real_idx] = e;
                    infos[real_idx] = v.info_;
                    effect_pts[real_idx] = true;

                    // compute score
                    double score = -gauss_d1_ * std::exp(-gauss_d2_ * e.dot(v.info_ * e) / 2);
                    if (score > 1.0 || score < 0.0 || std::isnan(score)) {
                        score = 0.0;
                    }
                    scores[real_idx] = score;
                } else {
                    effect_pts[real_idx] = false;
                }
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;

        Mat6d H = Mat6d::Zero();
        Vec6d err = Vec6d::Zero();

        double score_in_total = 0.0;
        for (int idx = 0; idx < effect_pts.size(); ++idx) {
            if (!effect_pts[idx]) {
                continue;
            }

            total_res += errors[idx].transpose() * infos[idx] * errors[idx];
            // chi2.emplace_back(errors[idx].transpose() * infos[idx] * errors[idx]);
            effective_num++;

            H += jacobians[idx].transpose() * infos[idx] * jacobians[idx];
            err += -jacobians[idx].transpose() * infos[idx] * errors[idx];
            score_in_total += scores[idx];
        }

        trans_likelihood_ = score_in_total / static_cast<double>(source_->points.size());

        if (effective_num < options_.min_effective_pts_) {
            LOG(WARNING) << "effective num too small: " << effective_num;
            return false;
        }

        Vec6d dx = H.inverse() * err;
        pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        if (options_.print_opti_progress) {
            LOG(INFO) << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num
                      << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm()
                      << ", dx: " << dx.transpose() << ", trans prob: " << trans_likelihood_;
        }

        // std::sort(chi2.begin(), chi2.end());
        // LOG(INFO) << "chi2 med: " << chi2[chi2.size() / 2] << ", .7: " << chi2[chi2.size() * 0.7]
        //           << ", .9: " << chi2[chi2.size() * 0.9] << ", max: " << chi2.back();

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            LOG(INFO) << "iter " << iter << " pose error: " << pose_error;
        }

        if (dx.norm() < options_.eps_) {
            LOG(INFO) << "converged, dx = " << dx.transpose();
            break;
        }
    }

    init_pose = pose;
    return true;
}

void Ndt3d::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    }
}

void Ndt3d::ComputeTransProbFactors() {
    const double gauss_c1 = 10.0 * (1 - options_.outlier_ratio_);
    const double gauss_c2 = options_.outlier_ratio_ / std::pow(options_.voxel_size_, 3);
    const double gauss_d3 = -std::log(gauss_c2);
    gauss_d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
    gauss_d2_ = -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) / gauss_d1_);
}

std::unordered_map<Ndt3d::KeyType, Ndt3d::VoxelData, hash_vec<3>> LoadNdtVoxels(const std::string& filename,
                                                                                double voxel_res) {
    std::unordered_map<Ndt3d::KeyType, Ndt3d::VoxelData, hash_vec<3>> voxels;

    std::ifstream ifs(filename);
    while (ifs.good()) {
        std::string line;
        std::getline(ifs, line);
        std::stringstream ss;
        ss << line;

        double mu_x, mu_y, mu_z;
        ss >> mu_x >> mu_y >> mu_z;

        double info_00, info_01, info_02, info_11, info_12, info_22;
        ss >> info_00 >> info_01 >> info_02 >> info_11 >> info_12 >> info_22;

        Ndt3d::VoxelData voxel;
        // mean
        voxel.mu_ << mu_x, mu_y, mu_z;
        // info
        voxel.info_(0, 0) = info_00;
        voxel.info_(0, 1) = info_01;
        voxel.info_(0, 2) = info_02;

        voxel.info_(1, 0) = info_01;
        voxel.info_(1, 1) = info_11;
        voxel.info_(1, 2) = info_12;

        voxel.info_(2, 0) = info_02;
        voxel.info_(2, 1) = info_12;
        voxel.info_(2, 2) = info_22;

        // cov
        voxel.sigma_ = voxel.info_.inverse();

        Ndt3d::KeyType key = (voxel.mu_ / voxel_res).cast<int>();
        voxels[key] = voxel;
    }

    return voxels;
}

}  // namespace sad