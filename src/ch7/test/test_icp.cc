//
// Created by xiang on 2022/7/7.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

#include "ch7/icp_3d.h"
#include "ch7/icp_3d_ceres.h"
#include "ch7/ndt_3d.h"
#include "common/point_cloud_utils.h"
#include "common/sys_utils.h"

DEFINE_string(source, "./data/ch7/EPFL/kneeling_lady_source.pcd", "第1个点云路径");
DEFINE_string(target, "./data/ch7/EPFL/kneeling_lady_target.pcd", "第2个点云路径");
DEFINE_string(ground_truth_file, "./data/ch7/EPFL/kneeling_lady_pose.txt", "真值Pose");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    // EPFL 雕像数据集：./ch7/EPFL/aquarius_{sourcd.pcd, target.pcd}，真值在对应目录的_pose.txt中
    // EPFL 模型比较精细，配准时应该采用较小的栅格

    std::ifstream fin(FLAGS_ground_truth_file);
    SE3 gt_pose;
    if (fin) {
        double tx, ty, tz, qw, qx, qy, qz;
        fin >> tx >> ty >> tz >> qw >> qx >> qy >> qz;
        fin.close();
        gt_pose = SE3(Quatd(qw, qx, qy, qz), Vec3d(tx, ty, tz));
    }

    sad::CloudPtr source(new sad::PointCloudType), target(new sad::PointCloudType);
    pcl::io::loadPCDFile(fLS::FLAGS_source, *source);
    pcl::io::loadPCDFile(fLS::FLAGS_target, *target);

    bool success;

    LOG(INFO) << "###### ICP P2P ######";
    sad::evaluate_and_call(
        [&]() {
            sad::Icp3d icp;
            icp.SetSource(source);
            icp.SetTarget(target);
            icp.SetGroundTruth(gt_pose);
            SE3 pose;
            success = icp.AlignP2P(pose);
            if (success) {
                LOG(INFO) << "icp p2p align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << pose.translation().transpose();
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
                sad::SaveCloudToFile("./data/ch7/icp_trans.pcd", *source_trans);
            } else {
                LOG(ERROR) << "align failed.";
            }
        },
        "ICP P2P", 1);

    LOG(INFO) << "###### ICP P2P Ceres Auto Diff ######";
    sad::evaluate_and_call(
        [&]() {
            sad::Icp3dCeres icp;
            icp.SetSource(source);
            icp.SetTarget(target);
            icp.SetGroundTruth(gt_pose);
            SE3 pose;
            success = icp.AlignP2P(pose);
            if (success) {
                LOG(INFO) << "icp p2p align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << pose.translation().transpose();
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
                sad::SaveCloudToFile("./data/ch7/icp_ceres_auto_diff_trans.pcd", *source_trans);
            } else {
                LOG(ERROR) << "align failed.";
            }
        },
        "ICP P2P Ceres Auto Diff", 1);

    LOG(INFO) << "###### ICP P2P Ceres Analytic Diff ######";
    sad::evaluate_and_call(
        [&]() {
            sad::Icp3d::Options options;
            options.use_auto_diff = false;
            sad::Icp3dCeres icp(options);
            icp.SetSource(source);
            icp.SetTarget(target);
            icp.SetGroundTruth(gt_pose);
            SE3 pose;
            success = icp.AlignP2P(pose);
            if (success) {
                LOG(INFO) << "icp p2p align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << pose.translation().transpose();
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
                sad::SaveCloudToFile("./data/ch7/icp_ceres_analytic_diff_trans.pcd", *source_trans);
            } else {
                LOG(ERROR) << "align failed.";
            }
        },
        "ICP P2P Ceres Analytic Diff", 1);

    /// 点到面
    LOG(INFO) << "###### ICP P2Plane ######";
    sad::evaluate_and_call(
        [&]() {
            sad::Icp3d icp;
            icp.SetSource(source);
            icp.SetTarget(target);
            icp.SetGroundTruth(gt_pose);
            SE3 pose;
            success = icp.AlignP2Plane(pose);
            if (success) {
                LOG(INFO) << "icp p2plane align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << pose.translation().transpose();
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
                sad::SaveCloudToFile("./data/ch7/icp_plane_trans.pcd", *source_trans);
            } else {
                LOG(ERROR) << "align failed.";
            }
        },
        "ICP P2Plane", 1);

    LOG(INFO) << "###### ICP P2Plane Ceres Analytic Diff ######";
    sad::evaluate_and_call(
        [&]() {
            sad::Icp3d::Options options;
            options.use_auto_diff = false;
            sad::Icp3dCeres icp(options);
            icp.SetSource(source);
            icp.SetTarget(target);
            icp.SetGroundTruth(gt_pose);
            SE3 pose;
            success = icp.AlignP2Plane(pose);
            if (success) {
                LOG(INFO) << "icp p2plane align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << pose.translation().transpose();
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
                sad::SaveCloudToFile("./data/ch7/icp_plane_ceres_analytic_diff_trans.pcd", *source_trans);
            } else {
                LOG(ERROR) << "align failed.";
            }
        },
        "ICP p2plane Ceres Analytic Diff", 1);

    /// 点到线
    LOG(INFO) << "###### ICP P2Line ######";
    sad::evaluate_and_call(
        [&]() {
            sad::Icp3d icp;
            icp.SetSource(source);
            icp.SetTarget(target);
            icp.SetGroundTruth(gt_pose);
            SE3 pose;
            success = icp.AlignP2Line(pose);
            if (success) {
                LOG(INFO) << "icp p2line align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << pose.translation().transpose();
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
                sad::SaveCloudToFile("./data/ch7/icp_line_trans.pcd", *source_trans);
            } else {
                LOG(ERROR) << "align failed.";
            }
        },
        "ICP P2Line", 1);

    LOG(INFO) << "###### ICP P2Line Ceres Analytic Diff ######";
    sad::evaluate_and_call(
        [&]() {
            sad::Icp3d::Options options;
            options.use_auto_diff = false;
            sad::Icp3dCeres icp(options);
            icp.SetSource(source);
            icp.SetTarget(target);
            icp.SetGroundTruth(gt_pose);
            SE3 pose;
            success = icp.AlignP2Line(pose);
            if (success) {
                LOG(INFO) << "icp p2line align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << pose.translation().transpose();
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
                sad::SaveCloudToFile("./data/ch7/icp_line_ceres_analytic_diff_trans.pcd", *source_trans);
            } else {
                LOG(ERROR) << "align failed.";
            }
        },
        "ICP P2Line Ceres Analytic Diff", 1);

    /// 第７章的NDT
    LOG(INFO) << "###### NDT ######";
    sad::evaluate_and_call(
        [&]() {
            sad::Ndt3d::Options options;
            options.voxel_size_ = 0.5;
            options.remove_centroid_ = true;
            options.nearby_type_ = sad::Ndt3d::NearbyType::CENTER;
            sad::Ndt3d ndt(options);
            ndt.SetSource(source);
            ndt.SetTarget(target);
            ndt.SetGtPose(gt_pose);
            SE3 pose;
            success = ndt.AlignNdt(pose);
            if (success) {
                LOG(INFO) << "ndt align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose() << ", "
                          << pose.translation().transpose();
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
                sad::SaveCloudToFile("./data/ch7/ndt_trans.pcd", *source_trans);
            } else {
                LOG(ERROR) << "align failed.";
            }
        },
        "NDT", 1);

    /// PCL ICP 作为备选
    LOG(INFO) << "###### ICP PCL ######";
    sad::evaluate_and_call(
        [&]() {
            pcl::IterativeClosestPoint<sad::PointType, sad::PointType> icp_pcl;
            icp_pcl.setInputSource(source);
            icp_pcl.setInputTarget(target);
            sad::CloudPtr output_pcl(new sad::PointCloudType);
            icp_pcl.align(*output_pcl);
            SE3f T = SE3f(icp_pcl.getFinalTransformation());
            LOG(INFO) << "pose from icp pcl: " << T.so3().unit_quaternion().coeffs().transpose() << ", "
                      << T.translation().transpose();
            sad::SaveCloudToFile("./data/ch7/pcl_icp_trans.pcd", *output_pcl);

            // 计算GT pose差异
            double pose_error = (gt_pose.inverse() * T.cast<double>()).log().norm();
            LOG(INFO) << "ICP PCL pose error: " << pose_error;
        },
        "ICP PCL", 1);

    /// PCL NDT 作为备选
    LOG(INFO) << "###### NDT PCL ######";
    sad::evaluate_and_call(
        [&]() {
            pcl::NormalDistributionsTransform<sad::PointType, sad::PointType> ndt_pcl;
            ndt_pcl.setInputSource(source);
            ndt_pcl.setInputTarget(target);
            ndt_pcl.setResolution(0.5);
            sad::CloudPtr output_pcl(new sad::PointCloudType);
            ndt_pcl.align(*output_pcl);
            SE3f T = SE3f(ndt_pcl.getFinalTransformation());
            LOG(INFO) << "pose from ndt pcl: " << T.so3().unit_quaternion().coeffs().transpose() << ", "
                      << T.translation().transpose() << ', trans: ' << ndt_pcl.getTransformationProbability();
            sad::SaveCloudToFile("./data/ch7/pcl_ndt_trans.pcd", *output_pcl);
            LOG(INFO) << "score: " << ndt_pcl.getTransformationProbability();

            // 计算GT pose差异
            double pose_error = (gt_pose.inverse() * T.cast<double>()).log().norm();
            LOG(INFO) << "NDT PCL pose error: " << pose_error;
        },
        "NDT PCL", 1);

    return 0;
}