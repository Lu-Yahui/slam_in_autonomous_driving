#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Geometry>
#include "scan_context.h"

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;

DEFINE_string(source_file, "./data/scan_context/first.csv", "source");
DEFINE_string(target_file, "./data/scan_context/second.csv", "target");

scan_context::PointCloud<scan_context::SCPointType> LoadPcdAsPointCloud(const std::string& filename) {
    PointCloudType::Ptr pc(new PointCloudType);
    pcl::io::loadPCDFile(filename, *pc);

    scan_context::PointCloud<scan_context::SCPointType> out;
    out.points.reserve(pc->size());

    for (int i = 0; i < pc->size(); ++i) {
        out.points.emplace_back(scan_context::SCPointType{pc->points[i].x, pc->points[i].y, pc->points[i].z, 1.0});
    }

    return out;
}

scan_context::PointCloud<scan_context::SCPointType> LoadCsvAsPointCloud(const std::string& filename) {
    std::ifstream ifs(filename);
    scan_context::PointCloud<scan_context::SCPointType> out;
    while (ifs.good()) {
        std::string line;
        std::getline(ifs, line);
        std::stringstream ss;
        ss << line;
        scan_context::SCPointType p;
        ss >> p.x >> p.y >> p.z >> p.i;
        out.points.push_back(p);
    }
    return out;
}

void SaveScanContext(const Eigen::MatrixXd& sc, const std::string& filename) {
    std::ofstream ofs(filename);
    ofs << sc.rows() << " " << sc.cols() << "\n";
    for (int r = 0; r < sc.rows(); ++r) {
        for (int c = 0; c < sc.cols(); ++c) {
            ofs << sc(r, c) << " ";
        }
    }
    ofs << std::endl;
}

void TransformPointCloud(scan_context::PointCloud<scan_context::SCPointType>& pc) {
    double angle_rad = 5.0 * M_PI / 180.0;
    Eigen::Quaterniond q(std::cos(angle_rad), 0.0, 0.0, std::sin(angle_rad));
    Eigen::Vector3d t(0.2, 0.5, 0.0);
    for (auto& p : pc.points) {
        Eigen::Vector3d p_vec(p.x, p.y, p.z);
        Eigen::Vector3d p_trans = q * p_vec + t;
        p.x = p_trans.x();
        p.y = p_trans.y();
        p.z = p_trans.z();
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    auto pc1 = LoadCsvAsPointCloud(FLAGS_source_file);
    auto pc2 = LoadCsvAsPointCloud(FLAGS_target_file);

    TransformPointCloud(pc2);

    LOG(INFO) << "source: " << pc1.points.size() << ", target: " << pc2.points.size();

    scan_context::ScanContextManager sc_manager(scan_context::ScanContextManager::Options{});
    sc_manager.MakeAndSaveScancontextAndKeys(pc1);
    sc_manager.MakeAndSaveScancontextAndKeys(pc2);

    const auto& sc_list = sc_manager.GetPolarContexts();
    SaveScanContext(sc_list.front(), "./data/scan_context/first.sc.txt");
    SaveScanContext(sc_list.back(), "./data/scan_context/second.sc.txt");

    auto ret = sc_manager.DetectLoopClosureID();
    LOG(INFO) << "id: " << ret.first << ", confidence: " << ret.second;

    return 0;
}
