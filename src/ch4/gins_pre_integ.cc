//
// Created by xiang on 2021/7/19.
//

#include "ch4/gins_pre_integ.h"
#include "ch4/ceres_types.h"
#include "ch4/g2o_types.h"
#include "common/g2o_types.h"

#include <glog/logging.h>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <ceres/ceres.h>

#include <chrono>

namespace sad {

namespace {

void NavStateToRawData(const NavStated& state, double* p_v_ba_bg, double* quaternion) {
    const Eigen::Vector3d& p = state.p_;
    const Eigen::Quaterniond& q = state.R_.unit_quaternion();
    const Eigen::Vector3d& v = state.v_;
    const Eigen::Vector3d& ba = state.ba_;
    const Eigen::Vector3d& bg = state.bg_;

    // P
    p_v_ba_bg[0] = p.x();
    p_v_ba_bg[1] = p.y();
    p_v_ba_bg[2] = p.z();
    // V
    p_v_ba_bg[3] = v.x();
    p_v_ba_bg[4] = v.y();
    p_v_ba_bg[5] = v.z();
    // Ba
    p_v_ba_bg[6] = ba.x();
    p_v_ba_bg[7] = ba.y();
    p_v_ba_bg[8] = ba.z();
    // Bg
    p_v_ba_bg[9] = bg.x();
    p_v_ba_bg[10] = bg.y();
    p_v_ba_bg[11] = bg.z();

    // Q
    quaternion[0] = q.x();
    quaternion[1] = q.y();
    quaternion[2] = q.z();
    quaternion[3] = q.w();
}

}  // namespace

void GinsPreInteg::AddImu(const IMU& imu) {
    if (first_gnss_received_ && first_imu_received_) {
        pre_integ_->Integrate(imu, imu.timestamp_ - last_imu_.timestamp_);
    }

    first_imu_received_ = true;
    last_imu_ = imu;
    current_time_ = imu.timestamp_;
}

void GinsPreInteg::SetOptions(sad::GinsPreInteg::Options options) {
    double bg_rw2 = 1.0 / (options_.bias_gyro_var_ * options_.bias_gyro_var_);
    options_.bg_rw_info_.diagonal() << bg_rw2, bg_rw2, bg_rw2;
    double ba_rw2 = 1.0 / (options_.bias_acce_var_ * options_.bias_acce_var_);
    options_.ba_rw_info_.diagonal() << ba_rw2, ba_rw2, ba_rw2;

    double gp2 = options_.gnss_pos_noise_ * options_.gnss_pos_noise_;
    double gh2 = options_.gnss_height_noise_ * options_.gnss_height_noise_;
    double ga2 = options_.gnss_ang_noise_ * options_.gnss_ang_noise_;

    options_.gnss_info_.diagonal() << 1.0 / ga2, 1.0 / ga2, 1.0 / ga2, 1.0 / gp2, 1.0 / gp2, 1.0 / gh2;
    pre_integ_ = std::make_shared<IMUPreintegration>(options_.preinteg_options_);

    double o2 = 1.0 / (options_.odom_var_ * options_.odom_var_);
    options_.odom_info_.diagonal() << o2, o2, o2;

    prior_info_.block<6, 6>(9, 9) = Mat6d ::Identity() * 1e6;

    if (this_frame_) {
        this_frame_->bg_ = options_.preinteg_options_.init_bg_;
        this_frame_->ba_ = options_.preinteg_options_.init_ba_;
    }
}

void GinsPreInteg::AddGnss(const GNSS& gnss) {
    this_frame_ = std::make_shared<NavStated>(current_time_);
    this_gnss_ = gnss;

    if (!first_gnss_received_) {
        if (!gnss.heading_valid_) {
            // 要求首个GNSS必须有航向
            return;
        }

        // 首个gnss信号，将初始pose设置为该gnss信号
        this_frame_->timestamp_ = gnss.unix_time_;
        this_frame_->p_ = gnss.utm_pose_.translation();
        this_frame_->R_ = gnss.utm_pose_.so3();
        this_frame_->v_.setZero();
        this_frame_->bg_ = options_.preinteg_options_.init_bg_;
        this_frame_->ba_ = options_.preinteg_options_.init_ba_;

        pre_integ_ = std::make_shared<IMUPreintegration>(options_.preinteg_options_);

        last_frame_ = this_frame_;
        last_gnss_ = this_gnss_;
        first_gnss_received_ = true;
        current_time_ = gnss.unix_time_;
        return;
    }

    // 积分到GNSS时刻
    pre_integ_->Integrate(last_imu_, gnss.unix_time_ - current_time_);

    current_time_ = gnss.unix_time_;
    *this_frame_ = pre_integ_->Predict(*last_frame_, options_.gravity_);

    if (options_.use_ceres_solver) {
        OptimizeWithCeresSolver();
    } else {
        Optimize();
    }

    last_frame_ = this_frame_;
    last_gnss_ = this_gnss_;
}

void GinsPreInteg::AddOdom(const sad::Odom& odom) {
    last_odom_ = odom;
    last_odom_set_ = true;
}

void GinsPreInteg::OptimizeWithCeresSolver() {
    if (pre_integ_->dt_ < 1e-3) {
        return;
    }

    // last state
    double p_v_ba_bg_i[12];
    double q_i[4];
    NavStateToRawData(*last_frame_, p_v_ba_bg_i, q_i);

    // current state
    double p_v_ba_bg_j[12];
    double q_j[4];
    NavStateToRawData(*this_frame_, p_v_ba_bg_j, q_j);

    // build optimization problem
    ceres::Problem problem;
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.5);

    // IMU pre-integration constraint
    Eigen::Matrix<double, 15, 15> preint_sqrt_info;
    preint_sqrt_info.setZero();
    preint_sqrt_info.block<9, 9>(0, 0) = pre_integ_->cov_.inverse().llt().matrixL();
    preint_sqrt_info.block<3, 3>(9, 9) = options_.bg_rw_info_.llt().matrixL();
    preint_sqrt_info.block<3, 3>(12, 12) = options_.ba_rw_info_.llt().matrixL();
    auto preint_cost_func =
        ImuPreIntegrationError::Create(pre_integ_, preint_sqrt_info, options_.gravity_, pre_integ_->dt_);
    problem.AddResidualBlock(preint_cost_func, loss_function, p_v_ba_bg_i, q_i, p_v_ba_bg_j, q_j);

    // prior constraint
    Eigen::Matrix<double, 15, 15> prior_sqrt_info = prior_info_.llt().matrixL();
    auto prior_cost_func = StatePriorError::Create(last_frame_->p_, last_frame_->R_.unit_quaternion(), last_frame_->v_,
                                                   last_frame_->ba_, last_frame_->bg_, prior_sqrt_info);
    problem.AddResidualBlock(prior_cost_func, loss_function, p_v_ba_bg_i, q_i);

    // gnss constraint for last state
    Eigen::Matrix<double, 6, 6> last_gnss_sqrt_info = options_.gnss_info_.llt().matrixL();
    auto last_gnss_cost_func = GnssError::Create(last_gnss_.utm_pose_.translation(),
                                                 last_gnss_.utm_pose_.so3().unit_quaternion(), last_gnss_sqrt_info);
    problem.AddResidualBlock(last_gnss_cost_func, loss_function, p_v_ba_bg_i, q_i);

    // gnss constrain for current state
    Eigen::Matrix<double, 6, 6> this_gnss_sqrt_info = options_.gnss_info_.llt().matrixL();
    auto this_gnss_cost_func = GnssError::Create(this_gnss_.utm_pose_.translation(),
                                                 this_gnss_.utm_pose_.so3().unit_quaternion(), this_gnss_sqrt_info);
    problem.AddResidualBlock(this_gnss_cost_func, loss_function, p_v_ba_bg_j, q_j);

    // vel/odom constraint
    if (last_odom_set_) {
        // velocity obs
        double velo_l =
            options_.wheel_radius_ * last_odom_.left_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
        double velo_r =
            options_.wheel_radius_ * last_odom_.right_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
        double average_vel = 0.5 * (velo_l + velo_r);
        Eigen::Vector3d vel_odom(average_vel, 0.0, 0.0);
        Eigen::Vector3d vel_world = this_frame_->R_ * vel_odom;

        Eigen::Matrix3d vel_sqrt_info = options_.odom_info_.llt().matrixL();
        auto vel_cost_func = VelError::Create(vel_world, vel_sqrt_info);
        problem.AddResidualBlock(vel_cost_func, loss_function, p_v_ba_bg_j);

        last_odom_set_ = false;
    }

    problem.SetManifold(q_i, quaternion_manifold);
    problem.SetManifold(q_j, quaternion_manifold);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (options_.verbose_) {
        LOG(INFO) << "Full report: \n" << summary.FullReport();
    }

    // update result
    last_frame_->R_ = SO3(Eigen::Quaterniond(q_i[3], q_i[0], q_i[1], q_i[2]).toRotationMatrix());
    last_frame_->p_ = Eigen::Vector3d(p_v_ba_bg_i[0], p_v_ba_bg_i[1], p_v_ba_bg_i[2]);
    last_frame_->v_ = Eigen::Vector3d(p_v_ba_bg_i[3], p_v_ba_bg_i[4], p_v_ba_bg_i[5]);
    last_frame_->ba_ = Eigen::Vector3d(p_v_ba_bg_i[6], p_v_ba_bg_i[7], p_v_ba_bg_i[8]);
    last_frame_->bg_ = Eigen::Vector3d(p_v_ba_bg_i[9], p_v_ba_bg_i[10], p_v_ba_bg_i[11]);

    this_frame_->R_ = SO3(Eigen::Quaterniond(q_j[3], q_j[0], q_j[1], q_j[2]).toRotationMatrix());
    this_frame_->p_ = Eigen::Vector3d(p_v_ba_bg_j[0], p_v_ba_bg_j[1], p_v_ba_bg_j[2]);
    this_frame_->v_ = Eigen::Vector3d(p_v_ba_bg_j[3], p_v_ba_bg_j[4], p_v_ba_bg_j[5]);
    this_frame_->ba_ = Eigen::Vector3d(p_v_ba_bg_j[6], p_v_ba_bg_j[7], p_v_ba_bg_j[8]);
    this_frame_->bg_ = Eigen::Vector3d(p_v_ba_bg_j[9], p_v_ba_bg_j[10], p_v_ba_bg_j[11]);

    // 重置integ
    options_.preinteg_options_.init_bg_ = this_frame_->bg_;
    options_.preinteg_options_.init_ba_ = this_frame_->ba_;
    pre_integ_ = std::make_shared<IMUPreintegration>(options_.preinteg_options_);
}

void GinsPreInteg::Optimize() {
    if (pre_integ_->dt_ < 1e-3) {
        return;
    }

    using BlockSolverType = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // 上时刻顶点， pose, v, bg, ba
    auto v0_pose = new VertexPose();
    v0_pose->setId(0);
    v0_pose->setEstimate(last_frame_->GetSE3());
    optimizer.addVertex(v0_pose);

    auto v0_vel = new VertexVelocity();
    v0_vel->setId(1);
    v0_vel->setEstimate(last_frame_->v_);
    optimizer.addVertex(v0_vel);

    auto v0_bg = new VertexGyroBias();
    v0_bg->setId(2);
    v0_bg->setEstimate(last_frame_->bg_);
    optimizer.addVertex(v0_bg);

    auto v0_ba = new VertexAccBias();
    v0_ba->setId(3);
    v0_ba->setEstimate(last_frame_->ba_);
    optimizer.addVertex(v0_ba);

    // 本时刻顶点，pose, v, bg, ba
    auto v1_pose = new VertexPose();
    v1_pose->setId(4);
    v1_pose->setEstimate(this_frame_->GetSE3());
    optimizer.addVertex(v1_pose);

    auto v1_vel = new VertexVelocity();
    v1_vel->setId(5);
    v1_vel->setEstimate(this_frame_->v_);
    optimizer.addVertex(v1_vel);

    auto v1_bg = new VertexGyroBias();
    v1_bg->setId(6);
    v1_bg->setEstimate(this_frame_->bg_);
    optimizer.addVertex(v1_bg);

    auto v1_ba = new VertexAccBias();
    v1_ba->setId(7);
    v1_ba->setEstimate(this_frame_->ba_);
    optimizer.addVertex(v1_ba);

    // 预积分边
    auto edge_inertial = new EdgeInertial(pre_integ_, options_.gravity_);
    edge_inertial->setVertex(0, v0_pose);
    edge_inertial->setVertex(1, v0_vel);
    edge_inertial->setVertex(2, v0_bg);
    edge_inertial->setVertex(3, v0_ba);
    edge_inertial->setVertex(4, v1_pose);
    edge_inertial->setVertex(5, v1_vel);
    auto* rk = new g2o::RobustKernelHuber();
    rk->setDelta(200.0);
    edge_inertial->setRobustKernel(rk);
    optimizer.addEdge(edge_inertial);

    // 两个零偏随机游走
    auto* edge_gyro_rw = new EdgeGyroRW();
    edge_gyro_rw->setVertex(0, v0_bg);
    edge_gyro_rw->setVertex(1, v1_bg);
    edge_gyro_rw->setInformation(options_.bg_rw_info_);
    optimizer.addEdge(edge_gyro_rw);

    auto* edge_acc_rw = new EdgeAccRW();
    edge_acc_rw->setVertex(0, v0_ba);
    edge_acc_rw->setVertex(1, v1_ba);
    edge_acc_rw->setInformation(options_.ba_rw_info_);
    optimizer.addEdge(edge_acc_rw);

    // 上时刻先验
    auto* edge_prior = new EdgePriorPoseNavState(*last_frame_, prior_info_);
    edge_prior->setVertex(0, v0_pose);
    edge_prior->setVertex(1, v0_vel);
    edge_prior->setVertex(2, v0_bg);
    edge_prior->setVertex(3, v0_ba);
    optimizer.addEdge(edge_prior);

    // GNSS边
    auto edge_gnss0 = new EdgeGNSS(v0_pose, last_gnss_.utm_pose_);
    edge_gnss0->setInformation(options_.gnss_info_);
    optimizer.addEdge(edge_gnss0);

    auto edge_gnss1 = new EdgeGNSS(v1_pose, this_gnss_.utm_pose_);
    edge_gnss1->setInformation(options_.gnss_info_);
    optimizer.addEdge(edge_gnss1);

    // Odom边
    EdgeEncoder3D* edge_odom = nullptr;
    Vec3d vel_world = Vec3d::Zero();
    Vec3d vel_odom = Vec3d::Zero();
    if (last_odom_set_) {
        // velocity obs
        double velo_l =
            options_.wheel_radius_ * last_odom_.left_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
        double velo_r =
            options_.wheel_radius_ * last_odom_.right_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
        double average_vel = 0.5 * (velo_l + velo_r);
        vel_odom = Vec3d(average_vel, 0.0, 0.0);
        vel_world = this_frame_->R_ * vel_odom;

        edge_odom = new EdgeEncoder3D(v1_vel, vel_world);
        edge_odom->setInformation(options_.odom_info_);
        optimizer.addEdge(edge_odom);

        // 重置odom数据到达标志位，等待最新的odom数据
        last_odom_set_ = false;
    }

    optimizer.setVerbose(options_.verbose_);
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    if (options_.verbose_) {
        // 获取结果，统计各类误差
        LOG(INFO) << "chi2/error: ";
        LOG(INFO) << "preintegration: " << edge_inertial->chi2() << "/" << edge_inertial->error().transpose();
        // LOG(INFO) << "gnss0: " << edge_gnss0->chi2() << ", " << edge_gnss0->error().transpose();
        LOG(INFO) << "gnss1: " << edge_gnss1->chi2() << ", " << edge_gnss1->error().transpose();
        LOG(INFO) << "bias: " << edge_gyro_rw->chi2() << "/" << edge_acc_rw->error().transpose();
        LOG(INFO) << "prior: " << edge_prior->chi2() << "/" << edge_prior->error().transpose();
        if (edge_odom) {
            LOG(INFO) << "body vel: " << (v1_pose->estimate().so3().inverse() * v1_vel->estimate()).transpose();
            LOG(INFO) << "meas: " << vel_odom.transpose();
            LOG(INFO) << "odom: " << edge_odom->chi2() << "/" << edge_odom->error().transpose();
        }
    }

    last_frame_->R_ = v0_pose->estimate().so3();
    last_frame_->p_ = v0_pose->estimate().translation();
    last_frame_->v_ = v0_vel->estimate();
    last_frame_->bg_ = v0_bg->estimate();
    last_frame_->ba_ = v0_ba->estimate();

    this_frame_->R_ = v1_pose->estimate().so3();
    this_frame_->p_ = v1_pose->estimate().translation();
    this_frame_->v_ = v1_vel->estimate();
    this_frame_->bg_ = v1_bg->estimate();
    this_frame_->ba_ = v1_ba->estimate();

    // 重置integ
    options_.preinteg_options_.init_bg_ = this_frame_->bg_;
    options_.preinteg_options_.init_ba_ = this_frame_->ba_;
    pre_integ_ = std::make_shared<IMUPreintegration>(options_.preinteg_options_);
}

NavStated GinsPreInteg::GetState() const {
    if (this_frame_ == nullptr) {
        return {};
    }

    if (pre_integ_ == nullptr) {
        return *this_frame_;
    }

    return pre_integ_->Predict(*this_frame_, options_.gravity_);
}

}  // namespace sad