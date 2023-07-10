//
// Created by xiang on 22-11-10.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ch8/lio-iekf/lio_iekf.h"
#include "common/io_utils.h"
#include "common/sys_utils.h"
#include "common/timer/timer.h"

DEFINE_string(bag_path, "dataset/sad/nclt/20130110.bag", "path to rosbag");
DEFINE_string(dataset_type, "NCLT", "NCLT/ULHK/UTBM/AVIA");                 // 数据集类型
DEFINE_string(config, "config/velodyne_nclt.yaml", "path of config yaml");  // 配置文件类型
DEFINE_bool(display_map, true, "display map?");
DEFINE_uint32(alignment, 1U, "alignment type, 1-NDT, 2-P2P ICP, 3-P2Line ICP, 4-P2Plane ICP, 5-LoamLike");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path, sad::Str2DatasetType(FLAGS_dataset_type));

    sad::LioIEKF::Options options{};
    options.alignment_ = static_cast<sad::LioIEKF::Alignment>(FLAGS_alignment);
    sad::LioIEKF lio(options);
    lio.Init(FLAGS_config);

    LOG(INFO) << "LIO alignment type: " << FLAGS_alignment
              << " (1-NDT, 2-P2P ICP, 3-P2Line ICP, 4-P2Plane ICP, 5-LoamLike)";

    rosbag_io
        .AddAutoPointCloudHandle([&](sensor_msgs::PointCloud2::Ptr cloud) -> bool {
            sad::common::Timer::Evaluate([&]() { lio.PCLCallBack(cloud); },
                                         "IEKF lio alignment-" + std::to_string(FLAGS_alignment));
            return true;
        })
        .AddLivoxHandle([&](const livox_ros_driver::CustomMsg::ConstPtr& msg) -> bool {
            sad::common::Timer::Evaluate([&]() { lio.LivoxPCLCallBack(msg); },
                                         "IEKF lio alignment-" + std::to_string(FLAGS_alignment));
            return true;
        })
        .AddImuHandle([&](IMUPtr imu) {
            lio.IMUCallBack(imu);
            return true;
        })
        .Go();

    lio.Finish();
    sad::common::Timer::PrintAll();
    LOG(INFO) << "done.";

    return 0;
}
