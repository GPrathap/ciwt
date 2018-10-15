//
// Created by root on 10/15/18.
//

#ifndef TRACY_NODE_TRACY_H
#define TRACY_NODE_TRACY_H
// C
#include <ctime>

// std
#include <iostream>
#include <memory>
#include <algorithm>
#include <list>
#include <chrono>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// boost
#include <boost/archive/binary_iarchive.hpp>

// scene segmentation
#include "include/scene_segmentation/scene_segmentation.h"
#include "include/scene_segmentation/utils_segmentation.h"
#include "include/scene_segmentation/multi_scale_quickshift.h"
#include "include/scene_segmentation/parameters_gop3D.h"

// tracking
#include "include/tracking/visualization.h"
#include "include/tracking/utils_tracking.h"
#include "src/sun_utils/detection.h"
#include "include/tracking/category_filter.h"

// utils
#include "src/sun_utils/utils_io.h"
#include "src/sun_utils/utils_visualization.h"
#include "src/sun_utils/utils_pointcloud.h"
#include "src/sun_utils/ground_model.h"
#include "src/sun_utils/utils_observations.h"
#include "src/sun_utils/datasets_dirty_utils.h"
#include "src/sun_utils/utils_bounding_box.h"
#include "src/sun_utils/utils_common.h"
#include "src/sun_utils/utils_filtering.h"

// CIWT
#include "src/CIWT/parameters_CIWT.h"
#include "src/CIWT/CIWT_tracker.h"
#include "src/CIWT/observation_fusion.h"
#include "src/CIWT/potential_functions.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <iostream>
#include <glob.h>
#include <vector>

#define MAX_PATH_LEN 500

namespace po = boost::program_options;

namespace tracy{

    class Tracy {

    public:
        Tracy() = default;
        void imageCallback(const sensor_msgs::ImageConstPtr& msg);
        bool RequestObjectProposals(int frame, po::variables_map &options_map,
                                           std::function<std::vector<GOT::segmentation::ObjectProposal>(po::variables_map &)> proposal_gen_fnc,
                                           std::vector<GOT::segmentation::ObjectProposal> &proposals_out,
                                           bool save_if_not_avalible=true);
        void VisualizeScene3D();
        bool ParseCommandArguments(const int argc, char **argv, po::variables_map &config_variables_map);

        std::string front_camera = "/apollo/sensor/camera/perception/depth_estimation/front_camera";
        // For convenience.
        typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;

        // We need those for the 3d visualizer thread.
        bool visualization_3d_update_flag;
        boost::mutex visualization_3d_update_mutex;
        PointCloudRGBA::Ptr visualization_3d_point_cloud;
        GOT::tracking::HypothesesVector visualization_3d_tracking_hypotheses;
        GOT::tracking::HypothesesVector visualization_3d_tracking_terminated_hypotheses;
        std::vector<GOT::tracking::Observation> visualization_observations;
        std::vector<GOT::segmentation::ObjectProposal> visualization_3d_proposals;
        SUN::utils::Camera visualization_3d_left_camera;
        GOT::tracking::Visualizer tracking_visualizer;

        // Paths
        std::string output_dir;
        std::string proposals_path;
        std::string tracking_mode_str;
        std::string viewer_3D_output_path;
        std::string config_parameters_file;
        std::string sequence_name;
        std::string dataset_name;

        // Application data.
        bool show_visualization_2d;
        bool show_visualization_3d;

        int start_frame;
        int end_frame;
        int debug_level;
        bool run_tracker;

        Eigen::Matrix4d egomotion = Eigen::Matrix4d::Identity();

    };
}

#endif
