#ifndef SAD_CH8_ICP_3D_INC_H
#define SAD_CH8_ICP_3D_INC_H

#include <deque>

#include "ch5/kdtree.h"
#include "common/eigen_types.h"
#include "common/point_types.h"

namespace sad {

class Icp3DIncBase {
   public:
    struct Options {
        int max_point_cloud_ = 3;
        double max_nn_distance_ = 1.0;
        double max_plane_distance_ = 0.05;
        double max_line_distance_ = 0.5;
    };

    explicit Icp3DIncBase(const Options& options);

    virtual void SetSource(CloudPtr source) { source_ = source; }

    virtual void AddCloud(CloudPtr cloud_world);

    virtual void ComputeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) = 0;

   protected:
    virtual void BuildKdTree();

   protected:
    Options options_;
    std::deque<CloudPtr> clouds_;
    CloudPtr source_{nullptr};
    std::unique_ptr<KdTree> kdtree_{nullptr};
    CloudPtr merged_cloud_world_{nullptr};
};

class Icp3DIncP2P : public Icp3DIncBase {
   public:
    explicit Icp3DIncP2P(const Options& options = Options{});

    void ComputeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) override;
};

class Icp3DIncP2Line : public Icp3DIncBase {
   public:
    explicit Icp3DIncP2Line(const Options& options = Options{});

    void ComputeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) override;
};

class Icp3DIncP2Plane : public Icp3DIncBase {
   public:
    explicit Icp3DIncP2Plane(const Options& options = Options{});

    void ComputeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr) override;
};

}  // namespace sad

#endif  // SAD_CH8_ICP_3D_INC_H