#ifndef SLAM_IN_AUTO_DRIVING_ICP_3D_CERES_H
#define SLAM_IN_AUTO_DRIVING_ICP_3D_CERES_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <execution>

#include "common/math_utils.h"
#include "icp_3d.h"

namespace sad {

template <typename T>
inline Eigen::Matrix<T, 3, 3> Hat(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> r;
    r << static_cast<T>(0.0), -v[2], v[1], v[2], static_cast<T>(0.0), -v[0], -v[1], v[0], static_cast<T>(0.0);
    return r;
}

template <typename T>
inline Eigen::Matrix<T, 3, 1> Vee(const Eigen::Matrix<T, 3, 3>& r) {}

template <typename T>
inline Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1>& v) {
    T theta = v.norm();
    Eigen::Matrix<T, 3, 1> n = v.normalized();
    T cos_theta = std::cos(theta);
    T sin_theta = std::sin(theta);

    Eigen::Matrix<T, 3, 3> r = cos_theta * Eigen::Matrix<T, 3, 3>::Identity() +
                               (static_cast<T>(1.0) - cos_theta) * n * n.transpose() + sin_theta * Hat(n);
    return r;
}

template <typename T>
inline Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3>& r) {
    Eigen::AngleAxis<T> rot_vec(r);
    return rot_vec.angle() * rot_vec.axis();
}

struct SE3PoseParameterization : public ceres::LocalParameterization {
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        const Eigen::Map<const Eigen::Vector3d> x_rot_vec(&x[0]);
        const Eigen::Map<const Eigen::Vector3d> x_trans(&x[3]);
        const Eigen::Map<const Eigen::Vector3d> delta_rot_vec(&delta[0]);
        const Eigen::Map<const Eigen::Vector3d> delta_trans(&delta[3]);

        Eigen::Map<Eigen::Vector3d> x_plus_delta_rot_vec(&x_plus_delta[0]);
        Eigen::Map<Eigen::Vector3d> x_plus_delta_trans(&x_plus_delta[3]);

        Eigen::Matrix3d delta_rot = Exp<double>(delta_rot_vec);
        Eigen::Matrix3d x_rot = Exp<double>(x_rot_vec);
        x_plus_delta_rot_vec = Log<double>(x_rot * delta_rot);
        x_plus_delta_trans = x_trans + delta_trans;

        return true;
    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const {
        ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
        return true;
    }

    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

struct PointToPointErrorAutoDiff {
    PointToPointErrorAutoDiff(const Eigen::Vector3d& source_input, const Eigen::Vector3d& target_input)
        : source(source_input), target(target_input) {}

    template <typename T>
    bool operator()(const T* const rt_ptr, T* residuals_ptr) const {
        T s[3] = {static_cast<T>(source.x()), static_cast<T>(source.y()), static_cast<T>(source.z())};

        T p[3];
        ceres::AngleAxisRotatePoint(rt_ptr, s, p);

        p[0] += rt_ptr[3];
        p[1] += rt_ptr[4];
        p[2] += rt_ptr[5];

        residuals_ptr[0] = p[0] - static_cast<T>(target.x());
        residuals_ptr[1] = p[1] - static_cast<T>(target.y());
        residuals_ptr[2] = p[2] - static_cast<T>(target.z());

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& source_input, const Eigen::Vector3d& target_input) {
        return new ceres::AutoDiffCostFunction<PointToPointErrorAutoDiff, 3, 6>(
            new PointToPointErrorAutoDiff(source_input, target_input));
    }

    Eigen::Vector3d source;
    Eigen::Vector3d target;
};

struct PointToPointErrorAnalyticDiff : public ceres::SizedCostFunction<3, 6> {
    PointToPointErrorAnalyticDiff(const Eigen::Vector3d& source_input, const Eigen::Vector3d& target_input)
        : source(source_input), target(target_input) {}

    bool Evaluate(double const* const* parameters, double* residuals_ptr, double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> r(&parameters[0][0]);
        Eigen::Map<const Eigen::Vector3d> t(&parameters[0][3]);

        Eigen::Matrix3d R = Exp<double>(r);
        Eigen::Vector3d p = R * source + t;
        Eigen::Map<Eigen::Vector3d> residuals(residuals_ptr);
        residuals = p - target;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian(jacobians[0]);
                jacobian.setZero();
                jacobian.block<3, 3>(0, 0) = -R * Hat(source);
                jacobian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
            }
        }
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& source_input, const Eigen::Vector3d& target_input) {
        return new PointToPointErrorAnalyticDiff(source_input, target_input);
    }

    Eigen::Vector3d source;
    Eigen::Vector3d target;
};

struct PointToLineErrorAnalyticDiff : public ceres::SizedCostFunction<3, 6> {
    PointToLineErrorAnalyticDiff(const Eigen::Vector3d& source_input, const Eigen::Vector3d& line_dir_input,
                                 const Eigen::Vector3d& line_origin_input)
        : source(source_input),
          source_hat(Hat(source)),
          line_dir(line_dir_input),
          line_origin(line_origin_input),
          line_dir_hat(Hat(line_dir_input)) {}

    bool Evaluate(double const* const* parameters, double* residuals_ptr, double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> r(&parameters[0][0]);
        Eigen::Map<const Eigen::Vector3d> t(&parameters[0][3]);

        Eigen::Matrix3d R = Exp<double>(r);
        Eigen::Vector3d p = R * source + t;
        Eigen::Map<Eigen::Vector3d> residuals(residuals_ptr);
        residuals = line_dir_hat * (p - line_origin);

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian(jacobians[0]);
                jacobian.setZero();
                jacobian.block<3, 3>(0, 0) = -line_dir_hat * R * source_hat;
                jacobian.block<3, 3>(0, 3) = line_dir_hat;
            }
        }
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& source_input, const Eigen::Vector3d& line_dir_input,
                                       const Eigen::Vector3d& line_origin_input) {
        return new PointToLineErrorAnalyticDiff(source_input, line_dir_input, line_origin_input);
    }

    const Eigen::Vector3d source;
    const Eigen::Matrix3d source_hat;
    const Eigen::Vector3d line_dir;
    const Eigen::Vector3d line_origin;
    const Eigen::Matrix3d line_dir_hat;
};

struct PointToPlaneErrorAnalyticDiff : public ceres::SizedCostFunction<1, 6> {
    PointToPlaneErrorAnalyticDiff(const Eigen::Vector3d& source_input, const Eigen::Vector4d& plane_input)
        : source(source_input), source_hat(Hat(source)), plane(plane_input) {}

    bool Evaluate(double const* const* parameters, double* residuals_ptr, double** jacobians) const override {
        Eigen::Map<const Eigen::Vector3d> r(&parameters[0][0]);
        Eigen::Map<const Eigen::Vector3d> t(&parameters[0][3]);

        Eigen::Matrix3d R = Exp<double>(r);
        Eigen::Vector3d p = R * source + t;
        residuals_ptr[0] = plane.head<3>().dot(p) + plane[3];

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jacobian(jacobians[0]);
                jacobian.setZero();
                jacobian.block<1, 3>(0, 0) = -plane.head<3>().transpose() * R * source_hat;
                jacobian.block<1, 3>(0, 3) = plane.head<3>().transpose();
            }
        }
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& source_input, const Eigen::Vector4d& plane_input) {
        return new PointToPlaneErrorAnalyticDiff(source_input, plane_input);
    }

    const Eigen::Vector3d source;
    const Eigen::Matrix3d source_hat;
    const Eigen::Vector4d plane;
};

class Icp3dCeres : public Icp3d {
   public:
    Icp3dCeres() {}

    Icp3dCeres(Icp3d::Options options) : Icp3d(options) {}

    bool AlignP2P(SE3& init_pose) override {
        assert(target_ != nullptr && source_ != nullptr);

        SE3 pose = init_pose;
        if (!options_.use_initial_translation_) {
            pose.translation() = target_center_ - source_center_;
        }

        std::vector<int> index(source_->points.size());
        for (int i = 0; i < index.size(); ++i) {
            index[i] = i;
        }

        for (uint32_t round = 0U; round < 5U; ++round) {
            // setup init pose for ceres
            double pose_ptr[6];
            Eigen::AngleAxisd rot_vec(pose.so3().unit_quaternion());
            Eigen::Map<Eigen::Vector3d> pose_rot(pose_ptr);
            pose_rot = rot_vec.angle() * rot_vec.axis();
            Eigen::Map<Eigen::Vector3d> pose_trans(pose_ptr);
            pose_trans = pose.translation();

            // add parameter block
            ceres::Problem problem;
            ceres::LocalParameterization* se3_parameterization = new SE3PoseParameterization();
            problem.AddParameterBlock(pose_ptr, 6, se3_parameterization);

            // source and target index pair, -1 if invalid
            std::vector<std::pair<int, int>> point_pair_indices(source_->points.size(), {-1, -1});
            // point association
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                auto q = ToVec3d(source_->points[idx]);
                Vec3d qs = pose * q;
                std::vector<int> nn;
                kdtree_->GetClosestPoint(ToPointType(qs), nn, 1);
                if (!nn.empty()) {
                    Vec3d p = ToVec3d(target_->points[nn[0]]);
                    double dis2 = (p - qs).squaredNorm();
                    if (dis2 > options_.max_nn_distance_) {
                        return;
                    }
                    point_pair_indices[idx] = {idx, nn[0]};
                }
            });

            // add point to point residual blocks
            for (const auto& index_pair : point_pair_indices) {
                const int source_index = index_pair.first;
                const int target_index = index_pair.second;
                if (source_index != -1 && target_index != -1) {
                    Eigen::Vector3d source_point(source_->points[source_index].x, source_->points[source_index].y,
                                                 source_->points[source_index].z);
                    Eigen::Vector3d target_point(target_->points[target_index].x, target_->points[target_index].y,
                                                 target_->points[target_index].z);

                    ceres::LossFunction* loss_function = nullptr;
                    if (options_.use_auto_diff) {
                        auto cost_func = PointToPointErrorAutoDiff::Create(source_point, target_point);
                        problem.AddResidualBlock(cost_func, loss_function, pose_ptr);
                    } else {
                        auto cost_func = PointToPointErrorAnalyticDiff::Create(source_point, target_point);
                        problem.AddResidualBlock(cost_func, loss_function, pose_ptr);
                    }
                }
            }

            ceres::Solver::Options options;
            options.max_num_iterations = 3;
            options.linear_solver_type = ceres::DENSE_QR;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // update pose
            pose.so3() = SO3(Exp<double>(pose_rot));
            pose.translation() = Eigen::Vector3d(pose_ptr[3], pose_ptr[4], pose_ptr[5]);

            if (gt_set_) {
                double pose_error = (gt_pose_.inverse() * pose).log().norm();
                LOG(INFO) << "round " << round << " pose error: " << pose_error;
            }
        }

        init_pose = pose;
        return true;
    }

    bool AlignP2Line(SE3& init_pose) override {
        assert(target_ != nullptr && source_ != nullptr);

        SE3 pose = init_pose;
        if (!options_.use_initial_translation_) {
            pose.translation() = target_center_ - source_center_;
        }

        std::vector<int> index(source_->points.size());
        for (int i = 0; i < index.size(); ++i) {
            index[i] = i;
        }

        for (uint32_t round = 0U; round < 5U; ++round) {
            // setup init pose for ceres
            double pose_ptr[6];
            Eigen::AngleAxisd rot_vec(pose.so3().unit_quaternion());
            Eigen::Map<Eigen::Vector3d> pose_rot(pose_ptr);
            pose_rot = rot_vec.angle() * rot_vec.axis();
            Eigen::Map<Eigen::Vector3d> pose_trans(pose_ptr);
            pose_trans = pose.translation();

            // add parameter block
            ceres::Problem problem;
            ceres::LocalParameterization* se3_parameterization = new SE3PoseParameterization();
            problem.AddParameterBlock(pose_ptr, 6, se3_parameterization);

            // source point index, line dir, line origin, -1 if invalid
            std::vector<std::tuple<int, Eigen::Vector3d, Eigen::Vector3d>> point_line_pairs(
                source_->points.size(), {-1, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
            // point to line association
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                auto q = ToVec3d(source_->points[idx]);
                Vec3d qs = pose * q;
                std::vector<int> nn;
                // look for 5 points to fit a line
                kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);
                if (nn.size() == 5) {
                    std::vector<Vec3d> nn_eigen;
                    for (int i = 0; i < 5; ++i) {
                        nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                    }

                    Vec3d line_dir, line_origin;
                    if (math::FitLine(nn_eigen, line_origin, line_dir, options_.max_line_distance_)) {
                        point_line_pairs[idx] = {idx, line_dir, line_origin};
                    }
                }
            });

            // add point to line residual blocks
            for (const auto& point_line_pair : point_line_pairs) {
                const auto& [source_index, line_dir, line_origin] = point_line_pair;
                if (source_index != -1) {
                    Eigen::Vector3d source_point(source_->points[source_index].x, source_->points[source_index].y,
                                                 source_->points[source_index].z);
                    ceres::LossFunction* loss_function = nullptr;
                    if (options_.use_auto_diff) {
                    } else {
                        auto cost_func = PointToLineErrorAnalyticDiff::Create(source_point, line_dir, line_origin);
                        problem.AddResidualBlock(cost_func, loss_function, pose_ptr);
                    }
                }
            }

            ceres::Solver::Options options;
            options.max_num_iterations = 3;
            options.linear_solver_type = ceres::DENSE_QR;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // update pose
            pose.so3() = SO3(Exp<double>(pose_rot));
            pose.translation() = Eigen::Vector3d(pose_ptr[3], pose_ptr[4], pose_ptr[5]);

            if (gt_set_) {
                double pose_error = (gt_pose_.inverse() * pose).log().norm();
                LOG(INFO) << "round " << round << " pose error: " << pose_error;
            }
        }

        init_pose = pose;
        return true;
    }

    bool AlignP2Plane(SE3& init_pose) override {
        assert(target_ != nullptr && source_ != nullptr);

        SE3 pose = init_pose;
        if (!options_.use_initial_translation_) {
            pose.translation() = target_center_ - source_center_;
        }

        std::vector<int> index(source_->points.size());
        for (int i = 0; i < index.size(); ++i) {
            index[i] = i;
        }

        for (uint32_t round = 0U; round < 5U; ++round) {
            // setup init pose for ceres
            double pose_ptr[6];
            Eigen::AngleAxisd rot_vec(pose.so3().unit_quaternion());
            Eigen::Map<Eigen::Vector3d> pose_rot(pose_ptr);
            pose_rot = rot_vec.angle() * rot_vec.axis();
            Eigen::Map<Eigen::Vector3d> pose_trans(pose_ptr);
            pose_trans = pose.translation();

            // add parameter block
            ceres::Problem problem;
            ceres::LocalParameterization* se3_parameterization = new SE3PoseParameterization();
            problem.AddParameterBlock(pose_ptr, 6, se3_parameterization);

            // source point index, plane, -1 if invalid
            std::vector<std::tuple<int, Eigen::Vector4d>> point_plane_pairs(source_->points.size(),
                                                                            {-1, Eigen::Vector4d::Zero()});
            // point to plane association
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
                auto q = ToVec3d(source_->points[idx]);
                Vec3d qs = pose * q;
                std::vector<int> nn;
                // look for 5 points to fit a plane
                kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);
                if (nn.size() > 3) {
                    // convert to eigen
                    std::vector<Vec3d> nn_eigen;
                    for (int i = 0; i < nn.size(); ++i) {
                        nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                    }

                    Vec4d plane;
                    if (math::FitPlane(nn_eigen, plane)) {
                        double dis = plane.head<3>().dot(qs) + plane[3];
                        if (std::abs(dis) < options_.max_plane_distance_) {
                            point_plane_pairs[idx] = {idx, plane};
                        }
                    }
                }
            });

            // add point to plane residual blocks
            for (const auto& point_plane_pair : point_plane_pairs) {
                const auto& [source_index, plane] = point_plane_pair;
                if (source_index != -1) {
                    Eigen::Vector3d source_point(source_->points[source_index].x, source_->points[source_index].y,
                                                 source_->points[source_index].z);
                    ceres::LossFunction* loss_function = nullptr;
                    if (options_.use_auto_diff) {
                    } else {
                        auto cost_func = PointToPlaneErrorAnalyticDiff::Create(source_point, plane);
                        problem.AddResidualBlock(cost_func, loss_function, pose_ptr);
                    }
                }
            }

            ceres::Solver::Options options;
            options.max_num_iterations = 3;
            options.linear_solver_type = ceres::DENSE_QR;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // update pose
            pose.so3() = SO3(Exp<double>(pose_rot));
            pose.translation() = Eigen::Vector3d(pose_ptr[3], pose_ptr[4], pose_ptr[5]);

            if (gt_set_) {
                double pose_error = (gt_pose_.inverse() * pose).log().norm();
                LOG(INFO) << "round " << round << " pose error: " << pose_error;
            }
        }

        init_pose = pose;
        return true;
    }
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_ICP_3D_CERES_H
