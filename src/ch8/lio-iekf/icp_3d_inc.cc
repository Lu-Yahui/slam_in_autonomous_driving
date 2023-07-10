#include "icp_3d_inc.h"
#include "common/math_utils.h"

#include <execution>

namespace sad {

Icp3DIncBase::Icp3DIncBase(const Options& options) : options_(options) {}

void Icp3DIncBase::AddCloud(CloudPtr cloud_world) {
    clouds_.push_back(cloud_world);
    if (clouds_.size() > options_.max_point_cloud_) {
        clouds_.pop_front();
    }
}

void Icp3DIncBase::BuildKdTree() {
    // merge clouds in local map
    CloudPtr merged_cloud_world(new PointCloudType);
    for (const auto& cloud : clouds_) {
        (*merged_cloud_world) += (*cloud);
    }
    merged_cloud_world_ = merged_cloud_world;

    LOG(INFO) << "Total points: " << merged_cloud_world->points.size();

    kdtree_ = std::make_unique<KdTree>();
    kdtree_->BuildTree(merged_cloud_world);
    kdtree_->SetEnableANN();
}

Icp3DIncP2P::Icp3DIncP2P(const Options& options) : Icp3DIncBase(options) {}

void Icp3DIncP2P::ComputeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) {
    // build target kd tree
    BuildKdTree();

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    int total_size = index.size();

    std::vector<bool> effect_pts(total_size, false);
    std::vector<Eigen::Matrix<double, 3, 18>> jacobians(total_size);
    std::vector<Vec3d> errors(total_size);

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
        auto q = ToVec3d(source_->points[idx]);
        Vec3d qs = pose * q;

        std::vector<int> nn;
        kdtree_->GetClosestPoint(ToPointType(qs), nn, 1);

        if (!nn.empty()) {
            // convert to eigen
            Vec3d p = ToVec3d(merged_cloud_world_->points[nn[0]]);
            double dis2 = (p - qs).squaredNorm();
            if (dis2 > options_.max_nn_distance_) {
                effect_pts[idx] = false;
                return;
            }

            effect_pts[idx] = true;

            // build residual and jacobians
            Vec3d e = qs - p;
            Eigen::Matrix<double, 3, 18> J;
            J.setZero();
            J.block<3, 3>(0, 0) = Mat3d::Identity();
            J.block<3, 3>(0, 6) = -pose.so3().matrix() * SO3::hat(q);

            jacobians[idx] = J;
            errors[idx] = e;
        } else {
            effect_pts[idx] = false;
        }
    });

    int effective_num = 0;
    HTVH.setZero();
    HTVr.setZero();
    for (int idx = 0; idx < effect_pts.size(); ++idx) {
        if (!effect_pts[idx]) {
            continue;
        }

        HTVH += jacobians[idx].transpose() * jacobians[idx];
        HTVr += -jacobians[idx].transpose() * errors[idx];

        effective_num++;
    }

    LOG(INFO) << "effective: " << effective_num;
}

Icp3DIncP2Line::Icp3DIncP2Line(const Options& options) : Icp3DIncBase(options) {}

void Icp3DIncP2Line::ComputeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) {
    // build target kd tree
    BuildKdTree();

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    int total_size = index.size();

    std::vector<bool> effect_pts(total_size, false);
    std::vector<Eigen::Matrix<double, 3, 18>> jacobians(total_size);
    std::vector<Vec3d> errors(total_size);

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
        auto q = ToVec3d(source_->points[idx]);
        Vec3d qs = pose * q;

        std::vector<int> nn;
        kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);

        if (nn.size() == 5) {
            // convert to eigen
            std::vector<Vec3d> nn_eigen;
            for (int i = 0; i < 5; ++i) {
                nn_eigen.emplace_back(ToVec3d(merged_cloud_world_->points[nn[i]]));
            }

            Vec3d d, p0;
            if (!math::FitLine(nn_eigen, p0, d, options_.max_line_distance_)) {
                effect_pts[idx] = false;
                return;
            }

            Vec3d err = SO3::hat(d) * (qs - p0);

            if (err.norm() > options_.max_line_distance_) {
                effect_pts[idx] = false;
                return;
            }

            effect_pts[idx] = true;

            // build residual and jacobians
            Eigen::Matrix<double, 3, 18> J;
            J.setZero();
            J.block<3, 3>(0, 0) = SO3::hat(d);
            J.block<3, 3>(0, 6) = -SO3::hat(d) * pose.so3().matrix() * SO3::hat(q);

            jacobians[idx] = J;
            errors[idx] = err;
        } else {
            effect_pts[idx] = false;
        }
    });

    int effective_num = 0;
    HTVH.setZero();
    HTVr.setZero();
    for (int idx = 0; idx < effect_pts.size(); ++idx) {
        if (!effect_pts[idx]) {
            continue;
        }

        HTVH += jacobians[idx].transpose() * jacobians[idx];
        HTVr += -jacobians[idx].transpose() * errors[idx];

        effective_num++;
    }

    LOG(INFO) << "effective: " << effective_num;
}

Icp3DIncP2Plane::Icp3DIncP2Plane(const Options& options) : Icp3DIncBase(options) {}

void Icp3DIncP2Plane::ComputeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) {
    // build target kd tree
    BuildKdTree();

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    int total_size = index.size();

    std::vector<bool> effect_pts(total_size, false);
    std::vector<Eigen::Matrix<double, 1, 18>> jacobians(total_size);
    std::vector<double> errors(total_size);

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
        auto q = ToVec3d(source_->points[idx]);
        Vec3d qs = pose * q;

        std::vector<int> nn;
        kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);

        if (nn.size() > 3) {
            // convert to eigen
            std::vector<Vec3d> nn_eigen;
            for (int i = 0; i < nn.size(); ++i) {
                nn_eigen.emplace_back(ToVec3d(merged_cloud_world_->points[nn[i]]));
            }

            Vec4d n;
            if (!math::FitPlane(nn_eigen, n)) {
                effect_pts[idx] = false;
                return;
            }

            double dis = n.head<3>().dot(qs) + n[3];
            if (std::abs(dis) > options_.max_plane_distance_) {
                effect_pts[idx] = false;
                return;
            }

            effect_pts[idx] = true;

            // build residual
            Eigen::Matrix<double, 1, 18> J;
            J.setZero();
            J.block<1, 3>(0, 0) = n.head<3>().transpose();
            J.block<1, 3>(0, 6) = -n.head<3>().transpose() * pose.so3().matrix() * SO3::hat(q);

            jacobians[idx] = J;
            errors[idx] = dis;
        } else {
            effect_pts[idx] = false;
        }
    });

    int effective_num = 0;
    HTVH.setZero();
    HTVr.setZero();
    for (int idx = 0; idx < effect_pts.size(); ++idx) {
        if (!effect_pts[idx]) {
            continue;
        }

        HTVH += jacobians[idx].transpose() * jacobians[idx];
        HTVr += -jacobians[idx].transpose() * errors[idx];

        effective_num++;
    }

    LOG(INFO) << "effective: " << effective_num;
}

}  // namespace sad
