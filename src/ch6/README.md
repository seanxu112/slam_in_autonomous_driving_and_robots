
P1:
点到点的ICP

边的定义，由于测量值不应该改变，所以将测量值直接放在边的构造函数中：

```
class EdgeICP_P2P : public g2o::BaseUnaryEdge<2, Vec2d, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeICP_P2P() {}
    EdgeICP_P2P(double rho_, double r_, Vec2d q_i_): rho(rho_), r(r_), q_i(q_i_){}

    void computeError() override {
        VertexSE2* current_pose = (VertexSE2*)_vertices[0];
        double x_ = current_pose->estimate().translation()[0];
        double y_ = current_pose->estimate().translation()[1];
        double theta_ = current_pose->estimate().so2().log();
        Vec2d p_i = {x_ + r * std::cos(rho+theta_), y_ + r * std::sin(rho+theta_)};
        _error = p_i - q_i;
    }

    // TODO jacobian
    void linearizeOplus() override {
        VertexSE2* current_pose = (VertexSE2*)_vertices[0];
        double theta_ = current_pose->estimate().so2().log();
        _jacobianOplusXi << 1, 0, 0, 1, -r * std::sin(rho + theta_), r * std::cos(rho + theta_);
    }
    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
        double rho;
        double r;
        double theta_, x_, y_;
        Vec2d q_i;
};
```

借用了AlignGaussNewton和likelihood的部分函数，把有效的测量值转换成世界坐标的点后找NN，将NN的坐标和测量值构造边。所有边构造完之后进行优化， 比较奇怪的是g2o的求解器在for循环外面构造的话会让跑的速度变得很慢。

```
ICP的函数：

bool Icp2d::AlignGaussNewtonP2P(SE2& init_pose) {
    SE2 current_pose = init_pose;

    const double range_th = 15.0;  // 不考虑太远的scan，不准
    const double rk_delta = 0.8;
    const float max_dis2 = 0.01; 

    
    for (int iter = 0; iter < 3; iter++){
        using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
        using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        auto* v = new VertexSE2();
        v->setId(0);
        v->setEstimate(current_pose);
        optimizer.addVertex(v);

    // 遍历source
        for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
            double r = source_scan_->ranges[i];
            if (r < source_scan_->range_min || r > source_scan_->range_max) {
                continue;
            }

            if (r > range_th) {
                continue;
            }

            double angle = source_scan_->angle_min + i * source_scan_->angle_increment;
            if (angle < source_scan_->angle_min + 30 * M_PI / 180.0 || angle > source_scan_->angle_max - 30 * M_PI / 180.0) {
                continue;
            }

            Vec2d q_i;

            Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));
            Point2d pt;
            pt.x = pw.x();
            pt.y = pw.y();
            // 最近邻
            std::vector<int> nn_idx;
            std::vector<float> dis;
            kdtree_.nearestKSearch(pt, 1, nn_idx, dis);

            if (nn_idx.size() > 0 && dis[0] < max_dis2) {
                Vec2d qw = Vec2d(target_cloud_->points[nn_idx[0]].x, target_cloud_->points[nn_idx[0]].y);  

                auto e = new EdgeICP_P2P(angle, r, qw);
                e->setVertex(0, v);
                e->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
                auto rk = new g2o::RobustKernelHuber;
                rk->setDelta(rk_delta);
                e->setRobustKernel(rk);
                optimizer.addEdge(e);
            }
        }

        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(1);
        current_pose = v->estimate();
    }

    init_pose = current_pose;
    return true;

}

```

<img src="P1_p2p.png" />


P2L的实现：
边的定义：
和课件上的差别不多，用线的常量来构建残差和雅各比

```
class EdgeICP_P2L : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeICP_P2L() {}
    EdgeICP_P2L(double rho_, double r_, Vec3d abc_): rho(rho_), r(r_), abc(abc_){}

    void computeError() override {
        VertexSE2* current_pose = (VertexSE2*)_vertices[0];
        double x_ = current_pose->estimate().translation()[0];
        double y_ = current_pose->estimate().translation()[1];
        double theta_ = current_pose->estimate().so2().log();
        Vec2d p_i = {x_ + r * std::cos(rho+theta_), y_ + r * std::sin(rho+theta_)};
        _error[0] = p_i[0] * abc[0] + p_i[1] * abc[1] + abc[2];
    }

    // TODO jacobian
    void linearizeOplus() override {
        VertexSE2* current_pose = (VertexSE2*)_vertices[0];
        double theta_ = current_pose->estimate().so2().log();
        _jacobianOplusXi << abc[0], abc[1], -abc[0]*r*std::sin(theta_+rho) + abc[1]*r*std::cos(theta_+rho);
    }
    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
        double rho;
        double r;
        Vec3d abc;
};
```

ICP的实现：
和上面P2P的实现差不多，就是找测量值在世界坐标里的5NN，再用5NN做线拟合。得出来的常量构建边。

```

bool Icp2d::AlignGaussNewtonP2L(SE2& init_pose) {
    SE2 current_pose = init_pose;

    const double range_th = 15.0;  // 不考虑太远的scan，不准
    const double rk_delta = 0.8;
    const float max_dis = 0.3;  
    const int min_effect_pts = 20;  // 最小有效点数

    
    for (int iter = 0; iter < 3; iter++){
        using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
        using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        auto* v = new VertexSE2();
        v->setId(0);
        v->setEstimate(current_pose);
        optimizer.addVertex(v);

        int effective_num = 0;  // 有效点数

    // 遍历source
        for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
            double r = source_scan_->ranges[i];
            if (r < source_scan_->range_min || r > source_scan_->range_max) {
                continue;
            }

            if (r > range_th) {
                continue;
            }

            double angle = source_scan_->angle_min + i * source_scan_->angle_increment;
            if (angle < source_scan_->angle_min + 30 * M_PI / 180.0 || angle > source_scan_->angle_max - 30 * M_PI / 180.0) {
                continue;
            }

            Vec2d q_i;

            Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));
            Point2d pt;
            pt.x = pw.x();
            pt.y = pw.y();
            // 最近邻
            std::vector<int> nn_idx;
            std::vector<float> dis;
            kdtree_.nearestKSearch(pt, 5, nn_idx, dis);

            std::vector<Vec2d> effective_pts;  // 有效点
            for (int j = 0; j < nn_idx.size(); ++j) {
                if (dis[j] < max_dis) {
                    effective_pts.emplace_back(
                        Vec2d(target_cloud_->points[nn_idx[j]].x, target_cloud_->points[nn_idx[j]].y));
                }
            }

            if (effective_pts.size() < 3) {
                continue;
            }

            // 拟合直线，组装J、H和误差
            Vec3d line_coeffs;
            if (math::FitLine2D(effective_pts, line_coeffs)) {
                // std::cout << "line_coeffs[0]: " << line_coeffs[0] << std::endl;
                effective_num++;
                auto* e = new EdgeICP_P2L(angle, r, line_coeffs);
                e->setVertex(0, v);
                e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
                optimizer.addEdge(e);
            }
        }
        if (effective_num < min_effect_pts) {
            return false;
        }

        optimizer.setVerbose(false);
        optimizer.initializeOptimization();
        optimizer.optimize(1);
        current_pose = v->estimate();
    }

    init_pose = current_pose;
    return true;

}


```

<img src="P1_p2l.png" />

实现结果上来看和手动实现的差别不大。


P2
```
0230907 01:02:52.285648 17573 sys_utils.h:32] 方法 Grid 3D 单线程 平均调用时间/次数: 7.50573/10 毫秒.
I20230907 01:02:52.285670 17573 test_nn.cc:108] truth: 18779, esti: 8572
I20230907 01:02:52.351713 17573 test_nn.cc:134] precision: 0.911339, recall: 0.415997, fp: 760, fn: 10967
I20230907 01:02:52.351732 17573 test_nn.cc:213] ===================
I20230907 01:02:52.368994 17573 sys_utils.h:32] 方法 Grid 3D 多线程 平均调用时间/次数: 1.7251/10 毫秒.
I20230907 01:02:52.369032 17573 test_nn.cc:108] truth: 18779, esti: 18779
I20230907 01:02:52.484673 17573 test_nn.cc:134] precision: 0.911339, recall: 0.415997, fp: 760, fn: 10967
I20230907 01:02:52.484694 17573 test_nn.cc:219] ===================
I20230907 01:02:52.631883 17573 sys_utils.h:32] 方法 Grid 3D 18 体素 单线程 平均调用时间/次数: 14.7177/10 毫秒.
I20230907 01:02:52.631903 17573 test_nn.cc:108] truth: 18779, esti: 10070
I20230907 01:02:52.699419 17573 test_nn.cc:134] precision: 0.964846, recall: 0.517386, fp: 354, fn: 9063
I20230907 01:02:52.699427 17573 test_nn.cc:225] ===================
I20230907 01:02:52.730659 17573 sys_utils.h:32] 方法 Grid 3D 18 体素 多线程 平均调用时间/次数: 3.12215/10 毫秒.
I20230907 01:02:52.730824 17573 test_nn.cc:108] truth: 18779, esti: 18779
I20230907 01:02:52.831476 17573 test_nn.cc:134] precision: 0.964846, recall: 0.517386, fp: 354, fn: 9063
```
可以看到18体素的速度比6体素慢很多，但是准确率和recall都高了不少。

P2:

<img src="HW 4.png" />


P3:
将nanoflann.h放到文件夹里，再把nanoflann里的utils.h中的PointCloud放到test_nn.cc中（因为例子中是用这个）。
尝试了一下用gridnn的形式来搭nanoflann的实现。一开始把KDTreeSingleIndexAdaptor放在getClosestPoint函数里面，但是速度太慢了，
放在构建函数中后编译错误，应该是KDTreeSingleIndexAdaptor没有默认（无输入的）构建函数所以遇到了一些问题，最后直接在test_nn.cc中实现了nanoflann

一开始使用了findNeighbors函数来找，但是在实现的时候没有找到5NN的实现方法，最后用了knnSearch 来做5NN

```
TEST(CH5_TEST, NANOFLANN_KNN) {
    sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
    pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
    pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

    if (first->empty() || second->empty()) {
        LOG(ERROR) << "cannot load cloud";
        FAIL();
    }

    // voxel grid 至 0.05
    sad::VoxelGrid(first);
    sad::VoxelGrid(second);

    PointCloud_flann<float> first_cloud;
    first_cloud.pts.resize(first->size());
    std::vector<size_t> num_index(first->size());
    std::for_each(num_index.begin(), num_index.end(), [idx = 0](size_t& i) mutable { i = idx++; });

    std::for_each(num_index.begin(), num_index.end(), [&first_cloud, &first, this](const size_t& idx) {
        auto pt = first->points[idx];
        first_cloud.pts[idx].x = pt.x;
        first_cloud.pts[idx].y = pt.y;
        first_cloud.pts[idx].z = pt.z;
    });

    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud_flann<float>>,
        PointCloud_flann<float>, 3>;
    my_kd_tree_t flann_knn(3, first_cloud, {10});
    flann_knn.buildIndex();

    // 比较 bfnn
    std::vector<std::pair<size_t, size_t>> true_matches;
    sad::bfnn_cloud_mt_k(first, second, true_matches);

    // 对第2个点云执行knn
    std::vector<std::pair<size_t, size_t>> matches;

    matches.clear();
    
    std::vector<size_t> index(second->size());

    matches.resize(index.size() * 5);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::for_each(index.begin(), index.end(), [idx = 0](size_t& i) mutable { i = idx++; });
    std::for_each(index.begin(), index.end(), [this, &matches, &second, &flann_knn](const size_t& idx) {
        size_t cp_idx;
        auto pt = second->points[idx];
        float query_pt[3] = {pt.x, pt.y, pt.z};
        size_t                num_results = 5;              // 最近的5个点
        std::vector<uint32_t> ret_index(num_results);       // 返回的点的索引
        std::vector<float>    out_dist_sqr(num_results);    // 返回的点的距离

        num_results = flann_knn.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        for (int i = 0; i < ret_index.size(); ++i) {
            matches[idx * 5 + i].first = ret_index[i];
            matches[idx * 5 + i].second = idx;
        }

    });
    auto t2 = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;

    LOG(INFO) << "方法 " << "NANOFLANN Kd Tree 5NN" << " 平均调用时间/次数: " << total_time << "/1 毫秒.";
    EvaluateMatches(true_matches, matches);

    LOG(INFO) << "done.";

    SUCCEED();
}
```

跑出来的结果：

```
I20230907 14:56:45.584944  7577 test_nn.cc:473] 方法 NANOFLANN Kd Tree 5NN 平均调用时间/次数: 13.4338/1 毫秒.
I20230907 14:56:45.584970  7577 test_nn.cc:108] truth: 93895, esti: 93895
I20230907 14:56:48.908072  7577 test_nn.cc:134] precision: 1, recall: 1, fp: 0, fn: 0
I20230907 14:56:48.908098  7577 test_nn.cc:476] done.
```

在我的电脑上其他算法的速度：


```
I20230907 14:56:25.992892  7577 sys_utils.h:32] 方法 Kd Tree build 平均调用时间/次数: 6.93181/1 毫秒.
I20230907 14:56:25.992902  7577 test_nn.cc:284] Kd tree leaves: 18869, points: 18869
I20230907 14:56:28.940630  7577 sys_utils.h:32] 方法 Kd Tree 5NN 多线程 平均调用时间/次数: 5.66564/1 毫秒.
I20230907 14:56:28.940672  7577 test_nn.cc:108] truth: 93895, esti: 93895
I20230907 14:56:32.387584  7577 test_nn.cc:134] precision: 1, recall: 1, fp: 0, fn: 0
I20230907 14:56:32.387611  7577 test_nn.cc:296] building kdtree pcl
I20230907 14:56:32.402770  7577 sys_utils.h:32] 方法 Kd Tree build 平均调用时间/次数: 15.1221/1 毫秒.
I20230907 14:56:32.402798  7577 test_nn.cc:301] searching pcl
I20230907 14:56:32.457382  7577 sys_utils.h:32] 方法 Kd Tree 5NN in PCL 平均调用时间/次数: 54.5544/1 毫秒.
I20230907 14:56:32.457669  7577 test_nn.cc:108] truth: 93895, esti: 93895
I20230907 14:56:36.010355  7577 test_nn.cc:134] precision: 1, recall: 1, fp: 0, fn: 0
I20230907 14:56:36.010380  7577 test_nn.cc:322] done.
[       OK ] CH5_TEST.KDTREE_KNN (10032 ms)
[ RUN      ] CH5_TEST.OCTREE_BASICS
I20230907 14:56:36.013893  7577 test_nn.cc:354] Octo tree leaves: 4, points: 4
[       OK ] CH5_TEST.OCTREE_BASICS (0 ms)
[ RUN      ] CH5_TEST.OCTREE_KNN
Failed to find match for field 'intensity'.
Failed to find match for field 'intensity'.
I20230907 14:56:36.052919  7577 sys_utils.h:32] 方法 Octo Tree build 平均调用时间/次数: 33.6464/1 毫秒.
I20230907 14:56:36.052945  7577 test_nn.cc:377] Octo tree leaves: 18869, points: 18869
I20230907 14:56:36.052949  7577 test_nn.cc:380] testing knn
I20230907 14:56:36.090013  7577 sys_utils.h:32] 方法 Octo Tree 5NN 多线程 平均调用时间/次数: 37.0516/1 毫秒.

```

可以看出，除了kd tree的速度比nanoflann快之外，其他的速度都比nanoflann慢一些。这几个算法的准确度和召回度也都在100%。
