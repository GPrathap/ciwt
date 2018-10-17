#ifndef __CAMERA_DEPTH_ESTIMATOR__
#define __CAMERA_DEPTH_ESTIMATOR__

#include <fstream>
#include <utility>
#include <vector>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <iostream>
#include <glob.h>
#include <vector>
#include <boost/assign/list_of.hpp>


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.



namespace depth {

    class Camera_Depth_Estimator {

    public:
        Camera_Depth_Estimator() {}

        virtual ~Camera_Depth_Estimator() {}

        tensorflow::string image = "tensorflow/examples/label_image/data/grace_hopper.jpg";
        tensorflow::string graph =
                "/root/monodepth/frozen_model.pb";
        tensorflow::string labels =
                "tensorflow/examples/label_image/data/imagenet_slim_labels.txt";
        std::string front_camera = "/apollo/sensor/camera/perception/depth_estimation/front_camera";
        tensorflow::int32 input_width = 512;
        tensorflow::int32 input_height = 256;
        float input_mean = 0;
        float input_std = 255;
        int next_image = 0;
        tensorflow::string input_layer = "split:0";
        tensorflow::string output_layer = "disparities/ExpandDims:0";
        bool self_test = false;
        tensorflow::string root_dir = "";
        std::vector<tensorflow::Flag> flag_list = {
                tensorflow::Flag("image", &image, "image to be processed"),
                tensorflow::Flag("graph", &graph, "graph to be executed"),
                tensorflow::Flag("labels", &labels, "name of file containing labels"),
                tensorflow::Flag("input_width", &input_width, "resize image to this width in pixels"),
                tensorflow::Flag("input_height", &input_height,
                                 "resize image to this height in pixels"),
                tensorflow::Flag("input_mean", &input_mean, "scale pixel values to this mean"),
                tensorflow::Flag("input_std", &input_std, "scale pixel values to this std deviation"),
                tensorflow::Flag("input_layer", &input_layer, "name of input layer"),
                tensorflow::Flag("output_layer", &output_layer, "name of output layer"),
                tensorflow::Flag("self_test", &self_test, "run a self test"),
                tensorflow::Flag("root_dir", &root_dir,
                                 "interpret image and graph file names relative to this directory"),
        };

        // First we load and initialize the model.
        std::unique_ptr<tensorflow::Session> session;

        tensorflow::Status ReadLabelsFile(const tensorflow::string &file_name, std::vector<tensorflow::string> *result,
                                          size_t *found_label_count);

        tensorflow::Status LoadGraph(const tensorflow::string &graph_file_name,
                                     std::unique_ptr<tensorflow::Session> *session);

        tensorflow::Status GetTopLabels(const std::vector<tensorflow::Tensor> &outputs, int how_many_labels,
                                        tensorflow::Tensor *indices, tensorflow::Tensor *scores);

        tensorflow::Status PrintTopLabels(const std::vector<tensorflow::Tensor> &outputs,
                                          const tensorflow::string &labels_file_name);

        tensorflow::Status CheckTopLabel(const std::vector<tensorflow::Tensor> &outputs, int expected,
                                         bool *is_expected);

        void getDisparityMap(cv::Mat source_img,  cv::Mat &disparity_map);

        template<typename T>
        std::vector<double> linspace(T start_in, T end_in, int num_in);


    };
};

#endif
