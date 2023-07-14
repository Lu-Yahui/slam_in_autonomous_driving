//
// Created by xiang on 22-12-20.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "ch7/ndt_3d.h"
#include "common/eigen_types.h"
#include "common/point_cloud_utils.h"
#include "keyframe.h"

DEFINE_string(map_path, "./data/ch9/", "导出数据的目录");
DEFINE_double(voxel_size, 0.1, "导出地图分辨率");
DEFINE_bool(split_ndt_map, true, "split and dump ndt map");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    using namespace sad;

    std::map<IdType, KFPtr> keyframes;
    if (!LoadKeyFrames("./data/ch9/keyframes.txt", keyframes)) {
        LOG(ERROR) << "failed to load keyframes";
        return 0;
    }

    std::map<Vec2i, CloudPtr, less_vec<2>> map_data;  // 以网格ID为索引的地图数据
    pcl::VoxelGrid<PointType> voxel_grid_filter;
    float resolution = FLAGS_voxel_size;
    voxel_grid_filter.setLeafSize(resolution, resolution, resolution);

    // 逻辑和dump map差不多，但每个点个查找它的网格ID，没有的话会创建
    for (auto& kfp : keyframes) {
        auto kf = kfp.second;
        kf->LoadScan("./data/ch9/");

        CloudPtr cloud_trans(new PointCloudType);
        pcl::transformPointCloud(*kf->cloud_, *cloud_trans, kf->opti_pose_2_.matrix());

        // voxel size
        CloudPtr kf_cloud_voxeled(new PointCloudType);
        voxel_grid_filter.setInputCloud(cloud_trans);
        voxel_grid_filter.filter(*kf_cloud_voxeled);

        LOG(INFO) << "building kf " << kf->id_ << " in " << keyframes.size();

        // add to grid
        for (const auto& pt : kf_cloud_voxeled->points) {
            int gx = floor((pt.x - 50.0) / 100);
            int gy = floor((pt.y - 50.0) / 100);
            Vec2i key(gx, gy);
            auto iter = map_data.find(key);
            if (iter == map_data.end()) {
                // create point cloud
                CloudPtr cloud(new PointCloudType);
                cloud->points.emplace_back(pt);
                cloud->is_dense = false;
                cloud->height = 1;
                map_data.emplace(key, cloud);
            } else {
                iter->second->points.emplace_back(pt);
            }
        }
    }

    // 存储点云和索引文件
    LOG(INFO) << "saving maps, grids: " << map_data.size();
    std::system("mkdir -p ./data/ch9/map_data/");
    std::system("rm -rf ./data/ch9/map_data/*");  // 清理一下文件夹
    std::ofstream fout("./data/ch9/map_data/map_index.txt");
    for (auto& dp : map_data) {
        fout << dp.first[0] << " " << dp.first[1] << std::endl;
        dp.second->width = dp.second->size();
        sad::VoxelGrid(dp.second, 0.1);

        sad::SaveCloudToFile(
            "./data/ch9/map_data/" + std::to_string(dp.first[0]) + "_" + std::to_string(dp.first[1]) + ".pcd",
            *dp.second);
    }
    fout.close();

    if (FLAGS_split_ndt_map) {
        std::vector<int> res = {10, 5, 4, 3, 1};
        for (const auto& r : res) {
            LOG(INFO) << "Loading whole NDT map, resolution: " << r;
            const auto& voxel_table = sad::LoadNdtVoxels("./data/ch9/ndt_map_" + std::to_string(r) + ".txt");
            std::map<Vec2i, std::vector<sad::Ndt3d::VoxelData>, less_vec<2>> ndt_map_data;
            for (const auto& kv : voxel_table) {
                const auto& v = kv.second;
                int gx = floor((v.mu_.x() - 50.0) / 100);
                int gy = floor((v.mu_.y() - 50.0) / 100);
                Vec2i key(gx, gy);
                ndt_map_data[key].push_back(v);
            }

            LOG(INFO) << "Saving NDT maps, grids: " << ndt_map_data.size();
            // setup folder
            std::string folder = "./data/ch9/ndt_map_data/res_" + std::to_string(r);
            std::system(std::string("mkdir -p " + folder).c_str());
            std::system(std::string("rm -rf " + folder + "/*").c_str());
            std::ofstream ofs(folder + "/ndt_map_index.txt");
            for (const auto& kv : ndt_map_data) {
                ofs << kv.first[0] << " " << kv.first[1] << std::endl;

                std::string tile_filename(folder + "/" + std::to_string(kv.first[0]) + "_" +
                                          std::to_string(kv.first[1]) + ".txt");
                std::ofstream tile_ofs(tile_filename);
                for (const auto& voxel_data : kv.second) {
                    // clang-format off
                    tile_ofs << voxel_data.mu_.x() << " " << voxel_data.mu_.y() << " " << voxel_data.mu_.z() << " "
                             << voxel_data.info_(0,0) << " " << voxel_data.info_(0,1) << " " << voxel_data.info_(0,2) << " "
                                                             << voxel_data.info_(1,1) << " " << voxel_data.info_(1,2) << " "
                                                                                             << voxel_data.info_(2,2) << "\n";
                    // clang-format on
                }
            }
        }
    }

    LOG(INFO) << "done.";
    return 0;
}
