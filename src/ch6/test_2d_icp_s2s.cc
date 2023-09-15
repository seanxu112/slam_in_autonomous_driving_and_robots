//
// Created by xiang on 2022/3/15.
//
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ch6/icp_2d.h"
#include "ch6/lidar_2d_utils.h"
#include "common/io_utils.h"
#include "common/point_types.h"
#include <pcl/sample_consensus/sac.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/filters/extract_indices.h>

DEFINE_string(bag_path, "./dataset/sad/2dmapping/floor1.bag", "数据包路径");
DEFINE_string(method, "point2point", "2d icp方法：point2point/point2plane");

void pruneCurrentScan(sad::CloudPtr& cloud_in, Eigen::VectorXf &coefficients, bool negative)
{
	// If negative = True, remove all points on the line based on the coefficients
	// If negative = False, remove all points outside of the line based on the coefficients
    pcl::ExtractIndices<sad::PointType> extractor;
	pcl::Indices inliers;
	pcl::PointIndices::Ptr inliers_pts (new pcl::PointIndices);
	std::size_t count_inliers;

	extractor.setNegative(negative);
	extractor.setInputCloud(cloud_in);

    std::vector< double > dists;

	pcl::SampleConsensusModelLine<sad::PointType>::Ptr line_model(new pcl::SampleConsensusModelLine<sad::PointType> (cloud_in)); //pcl::SampleConsensusModelPtr
	line_model->selectWithinDistance(coefficients, 0.3, inliers);
    // line_model->getDistancesToModel(coefficients, dists);
    // std::cout << "dists: ";
    // for (size_t idx = 0; idx < dists.size(); idx++)
    //     std::cout << dists[idx] << ", ";
    // std::cout << std::endl;
	inliers_pts->indices = inliers;
	extractor.setIndices(inliers_pts);
	extractor.filter(*cloud_in);

	return;
}

void getLineModel(sad::CloudPtr cloud_in, Eigen::VectorXf& coefficients)
{
	pcl::SampleConsensusModelLine<sad::PointType>::Ptr line_model(new pcl::SampleConsensusModelLine<sad::PointType> (cloud_in)); //pcl::SampleConsensusModelPtr
	// ransac_algo.setSampleConsensusModel(line_model);

	line_model->setInputCloud(cloud_in);
    pcl::RandomSampleConsensus<sad::PointType> ransac_algo(line_model);
	ransac_algo.setDistanceThreshold(0.3);
	ransac_algo.computeModel();
    
	ransac_algo.getModelCoefficients(coefficients);
    // std::cout << "slopes: ";
    // for (size_t idx = 0; idx < coefficients.size(); idx++)
    //     std::cout << coefficients[idx] << ", ";
    // std::cout << std::endl;
	return; 
}

std::size_t getInlierSize(sad::CloudPtr cloud_in, Eigen::VectorXf coefficients)
{
	pcl::SampleConsensusModelLine<sad::PointType>::Ptr line_model(new pcl::SampleConsensusModelLine<sad::PointType> (cloud_in)); //pcl::SampleConsensusModelPtr
	return line_model->countWithinDistance(coefficients, 0.3);
}


bool CheckScanQuality(Scan2d::Ptr scan)
{
    sad::CloudPtr cloud(new sad::PointCloudType());
    cloud->clear();
    int cloud_size = (scan->angle_max - 30 * M_PI / 180.0 - scan->angle_min - 30 * M_PI / 180.0) / scan->angle_increment+1;
    std::cout << "cloud_size: " << cloud_size << std::endl;
    cloud->resize(cloud_size);
    int idx = 0;
    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        if (scan->ranges[i] < scan->range_min || scan->ranges[i] > scan->range_max) {
            continue;
        }

        double real_angle = scan->angle_min + i * scan->angle_increment;
        double x = scan->ranges[i] * std::cos(real_angle);
        double y = scan->ranges[i] * std::sin(real_angle);

        // std::cout << "x: " << x << std::endl;
        // std::cout << "y: " << y << std::endl;

        if (real_angle < scan->angle_min + 30 * M_PI / 180.0 || real_angle > scan->angle_max - 30 * M_PI / 180.0) {
            continue;
        }

        cloud->points[idx].x = x;
        cloud->points[idx].y = y;
        cloud->points[idx].z = 0;
        idx++;
        // std::cout << "idx: " << idx << std::endl;
    }

    // std::cout << "************************" << std::endl;


    std::vector<Eigen::VectorXf> line_coeffs;

    while (cloud->size() > 70)
    {
        Eigen::VectorXf curr_coeff = Eigen::Matrix<float, 6, 1>::Zero();
        getLineModel(cloud, curr_coeff);
        size_t inlier_size = getInlierSize(cloud, curr_coeff);
        // std::cout << "inlier size: " << inlier_size << std::endl;
        
        if (inlier_size < 30)
            break;
        line_coeffs.push_back(curr_coeff);
        

        // std::cout << "cloud->size() before: " << cloud->size() << std::endl;

        pruneCurrentScan(cloud, curr_coeff, true);
        // std::cout << "cloud->size() after: " << cloud->size() << std::endl;
        // break;
    }

    std::vector<double> slopes;

    for (size_t i = 0; i < line_coeffs.size(); i++)
    {
        double dx = line_coeffs[i][3];
        double dy = line_coeffs[i][4];
        // std::cout << "slopes: " << dx << ", " << dy << std::endl;
        
        slopes.push_back(dy/dx);
    }

    sort(slopes.begin(), slopes.end());
    if (((slopes[slopes.size()-1] - slopes[0]) > 0.2) || cloud->size() > 200)
        std::cout << "The laser scan good" << std::endl;
    else
        std::cout << "The laser scan might be degenerate" << std::endl;

    // std::cout << "cloud->size() after: " << cloud->size() << std::endl;
    std::cout << "slopes: ";
    for (size_t idx = 0; idx < slopes.size(); idx++)
        std::cout << slopes[idx] << ", ";
    std::cout << std::endl;
    return true;
}

/// 测试从rosbag中读取2d scan并plot的结果
/// 通过选择method来确定使用点到点或点到面的ICP
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path);
    Scan2d::Ptr last_scan = nullptr, current_scan = nullptr;

    /// 我们将上一个scan与当前scan进行配准
    rosbag_io
        .AddScan2DHandle("/pavo_scan_bottom",
                         [&](Scan2d::Ptr scan) {
                             current_scan = scan;

                             if (last_scan == nullptr) {
                                 last_scan = current_scan;
                                 return true;
                             }

                             sad::Icp2d icp;
                             icp.SetTarget(last_scan);
                             icp.SetSource(current_scan);

                             bool scan_good = CheckScanQuality(current_scan);

                             SE2 pose;
                             if (fLS::FLAGS_method == "point2point") {
                                 icp.AlignGaussNewton(pose);
                             } else if (fLS::FLAGS_method == "point2plane") {
                                 icp.AlignGaussNewtonPoint2Plane(pose);
                             } else if (fLS::FLAGS_method == "point2point_g2o") {
                                 icp.AlignGaussNewtonP2P(pose);
                             } else if (fLS::FLAGS_method == "point2line_g2o") {
                                 icp.AlignGaussNewtonP2L(pose);
                             }
                            
                             cv::Mat image;
                             sad::Visualize2DScan(last_scan, SE2(), image, Vec3b(255, 0, 0));    // target是蓝的
                             sad::Visualize2DScan(current_scan, pose, image, Vec3b(0, 0, 255));  // source是红的
                             cv::imshow("scan", image);
                             cv::waitKey(20);

                             last_scan = current_scan;
                             return true;
                         })
        .Go();

    return 0;
}