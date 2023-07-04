#ifndef SLAM_IN_AUTO_DRIVING_CH4_CERES_TYPES_H
#define SLAM_IN_AUTO_DRIVING_CH4_CERES_TYPES_H

#include <ceres/ceres.h>
#include "ch4/imu_preintegration.h"
#include "common/eigen_types.h"

namespace sad {

template <typename T>
inline Eigen::Matrix<T, 3, 3> SkewSymmetric(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> m;
    m << static_cast<T>(0.0), -v[2], v[1], v[2], static_cast<T>(0.0), -v[0], -v[1], v[0], static_cast<T>(0.0);
    return m;
}

inline Eigen::Matrix3d exp(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    Eigen::Vector3d n = omega.normalized();
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    Eigen::Matrix3d so3 =
        cos_theta * Eigen::Matrix3d::Identity() + (1.0 - cos_theta) * n * n.transpose() + sin_theta * SkewSymmetric(n);
    return so3;
}

template <typename JetT>
inline Eigen::Matrix<JetT, 3, 3> Exp(const Eigen::Matrix<JetT, 3, 1>& omega) {
    Eigen::Vector3d v(omega.template x().a, omega.template y().a, omega.template z().a);
    Eigen::Matrix3d so3 = exp(v).template cast<JetT>();
    return so3;
}

struct GnssError {
    GnssError(const Eigen::Vector3d& p, const Eigen::Quaterniond& q, const Eigen::Matrix<double, 6, 6>& sqrt_info)
        : p_(p), q_(q), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p_v_ba_bg_ptr, const T* const q_ptr, T* residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(p_v_ba_bg_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q(q_ptr);

        Eigen::Quaternion<T> dq = q_.inverse().template cast<T>() * q;
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
        // R
        residuals.template block<3, 1>(0, 0) = T(2.0) * dq.vec();
        // P
        residuals.template block<3, 1>(3, 0) = p - p_.template cast<T>();

        residuals.applyOnTheLeft(sqrt_info_.template cast<T>());

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& p, const Eigen::Quaterniond& q,
                                       const Eigen::Matrix<double, 6, 6>& sqrt_info) {
        return new ceres::AutoDiffCostFunction<GnssError, 6, 12, 4>(new GnssError(p, q, sqrt_info));
    }

    const Eigen::Vector3d p_;
    const Eigen::Quaterniond q_;
    const Eigen::Matrix<double, 6, 6> sqrt_info_;
};

struct VelError {
    VelError(const Eigen::Vector3d& vel, const Eigen::Matrix3d& sqrt_info) : vel_(vel), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p_v_ba_bg_ptr, T* residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> vel(&p_v_ba_bg_ptr[3]);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals = vel - vel_.template cast<T>();
        residuals.applyOnTheLeft(sqrt_info_.template cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& vel, const Eigen::Matrix3d& sqrt_info) {
        return new ceres::AutoDiffCostFunction<VelError, 3, 12>(new VelError(vel, sqrt_info));
    }

    const Eigen::Vector3d vel_;
    const Eigen::Matrix3d sqrt_info_;
};

struct StatePriorError {
    StatePriorError(const Eigen::Vector3d& p, const Eigen::Quaterniond& q, const Eigen::Vector3d& v,
                    const Eigen::Vector3d& ba, const Eigen::Vector3d& bg,
                    const Eigen::Matrix<double, 15, 15>& sqrt_info)
        : p_(p), q_(q), v_(v), ba_(ba), bg_(bg), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p_v_ba_bg_ptr, const T* const q_ptr, T* residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> P(&p_v_ba_bg_ptr[0]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> V(&p_v_ba_bg_ptr[3]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Ba(&p_v_ba_bg_ptr[6]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bg(&p_v_ba_bg_ptr[9]);
        Eigen::Map<const Eigen::Quaternion<T>> Q(q_ptr);

        Eigen::Quaternion<T> dq = q_.inverse().template cast<T>() * Q;

        Eigen::Map<Eigen::Matrix<T, 15, 1>> residuals(residuals_ptr);
        // R
        residuals.template block<3, 1>(0, 0) = T(2.0) * dq.vec();
        // P
        residuals.template block<3, 1>(3, 0) = P - p_.template cast<T>();
        // V
        residuals.template block<3, 1>(6, 0) = V - v_.template cast<T>();
        // Bg
        residuals.template block<3, 1>(9, 0) = Bg - bg_.template cast<T>();
        // Ba
        residuals.template block<3, 1>(12, 0) = Ba - ba_.template cast<T>();

        residuals.applyOnTheLeft(sqrt_info_.template cast<T>());

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& p, const Eigen::Quaterniond& q, const Eigen::Vector3d& v,
                                       const Eigen::Vector3d& ba, const Eigen::Vector3d& bg,
                                       const Eigen::Matrix<double, 15, 15>& sqrt_info) {
        return new ceres::AutoDiffCostFunction<StatePriorError, 15, 12, 4>(
            new StatePriorError(p, q, v, ba, bg, sqrt_info));
    }

    const Eigen::Vector3d p_;
    const Eigen::Quaterniond q_;
    const Eigen::Vector3d v_;
    const Eigen::Vector3d ba_;
    const Eigen::Vector3d bg_;
    const Eigen::Matrix<double, 15, 15> sqrt_info_;
};

struct ImuPreIntegrationError {
    ImuPreIntegrationError(const std::shared_ptr<IMUPreintegration>& preint,
                           const Eigen::Matrix<double, 15, 15>& sqrt_information, const Eigen::Vector3d& gravity,
                           const double dt)
        : preint_(preint), sqrt_information_(sqrt_information), gravity_(gravity), dt_(dt) {}

    /**
     * @brief
     *
     * @tparam T
     * @param p_v_ba_bg_i_ptr p, v, ba, bj of state i
     * @param qi_ptr qx, qy, qz, qw of state i
     * @param p_v_ba_bg_j_ptr p, v, ba, bj of state j
     * @param qj_ptr qx, qy, qz, qw of state j
     * @param residuals_ptr R, P, V, Bg, Ba
     * @return true
     * @return false
     */
    template <typename T>
    bool operator()(const T* const p_v_ba_bg_i_ptr, const T* const qi_ptr, const T* const p_v_ba_bg_j_ptr,
                    const T* const qj_ptr, T* residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pi(&p_v_ba_bg_i_ptr[0]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Vi(&p_v_ba_bg_i_ptr[3]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bai(&p_v_ba_bg_i_ptr[6]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bgi(&p_v_ba_bg_i_ptr[9]);
        Eigen::Map<const Eigen::Quaternion<T>> Qi(qi_ptr);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pj(&p_v_ba_bg_j_ptr[0]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Vj(&p_v_ba_bg_j_ptr[3]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Baj(&p_v_ba_bg_j_ptr[6]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Bgj(&p_v_ba_bg_j_ptr[9]);
        Eigen::Map<const Eigen::Quaternion<T>> Qj(qj_ptr);

        // correct dq, preint_->GetDeltaRotation(Bgi);
        Eigen::Matrix<T, 3, 3> dR_dBg = preint_->dR_dbg_.template cast<T>();
        Eigen::Matrix<T, 3, 1> dBg = Bgi - preint_->bg_.template cast<T>();
        Eigen::Matrix<T, 3, 1> Bg_correction = dR_dBg * dBg;
        Eigen::Quaternion<T> quat_update(T(1.0), T(0.5) * Bg_correction[0], T(0.5) * Bg_correction[1],
                                         T(0.5) * Bg_correction[2]);
        Eigen::Quaternion<T> dq(preint_->dR_.unit_quaternion().template cast<T>() * quat_update);
        dq.normalize();

        // correct dv, preint_->GetDeltaVelocity(Bgi, Bai)
        Eigen::Matrix<T, 3, 3> dV_dBg = preint_->dV_dbg_.template cast<T>();
        Eigen::Matrix<T, 3, 3> dV_dBa = preint_->dV_dba_.template cast<T>();
        Eigen::Matrix<T, 3, 1> dBa = Bai - preint_->ba_.template cast<T>();
        Eigen::Matrix<T, 3, 1> dv = preint_->dv_.template cast<T>() + dV_dBg * dBg + dV_dBa * dBa;

        // correct dp, preint_->GetDeltaPosition(Bgi, Bai)
        Eigen::Matrix<T, 3, 3> dP_dBg = preint_->dP_dbg_.template cast<T>();
        Eigen::Matrix<T, 3, 3> dP_dBa = preint_->dP_dba_.template cast<T>();
        Eigen::Matrix<T, 3, 1> dp = preint_->dp_.template cast<T>() + dP_dBg * dBg + dP_dBa * dBa;

        Eigen::Matrix<T, 3, 3> RiT = Qi.inverse().toRotationMatrix();

        Eigen::Map<Eigen::Matrix<T, 15, 1>> residuals(residuals_ptr);
        // R
        residuals.template block<3, 1>(0, 0) = T(2.0) * (dq.inverse() * (Qi.inverse() * Qj)).vec();
        // P
        residuals.template block<3, 1>(3, 0) = RiT * (Pj - Pi - Vi * T(dt_) - T(0.5) * gravity_ * T(dt_) * T(dt_)) - dp;
        // V
        residuals.template block<3, 1>(6, 0) = RiT * (Vj - Vi - gravity_ * T(dt_)) - dv;
        // Bg
        residuals.template block<3, 1>(9, 0) = Bgi - Bgj;
        // Ba
        residuals.template block<3, 1>(12, 0) = Bai - Baj;

        residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const std::shared_ptr<IMUPreintegration>& preint,
                                       const Eigen::Matrix<double, 15, 15>& sqrt_information,
                                       const Eigen::Vector3d& gravity, const double dt) {
        return new ceres::AutoDiffCostFunction<ImuPreIntegrationError, 15, 12, 4, 12, 4>(
            new ImuPreIntegrationError(preint, sqrt_information, gravity, dt));
    }

    std::shared_ptr<IMUPreintegration> preint_;
    const Eigen::Matrix<double, 15, 15> sqrt_information_;
    const Eigen::Vector3d gravity_;
    const double dt_;
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_CH4_CERES_TYPES_H