
#include "../../include/depth_estimation/camera_depth_estimator.h"

namespace depth {

    tensorflow::Status
    Camera_Depth_Estimator::ReadLabelsFile(const tensorflow::string &file_name, std::vector<tensorflow::string> *result,
                                           size_t *found_label_count) {
        std::ifstream file(file_name);
        if (!file) {
            return tensorflow::errors::NotFound("Labels file ", file_name,
                                                " not found.");
        }
        result->clear();
        tensorflow::string line;
        while (std::getline(file, line)) {
            result->push_back(line);
        }
        *found_label_count = result->size();
        const int padding = 16;
        while (result->size() % padding) {
            result->emplace_back();
        }
        return tensorflow::Status::OK();
    }

    template<typename T>
    std::vector<double> Camera_Depth_Estimator::linspace(T start_in, T end_in, int num_in) {

        std::vector<double> linspaced;
        double start = static_cast<double>(start_in);
        double end_value = static_cast<double>(end_in);
        double num = static_cast<double>(num_in);
        if (num == 0) { return linspaced; }
        if (num == 1) {
            linspaced.push_back(start);
            return linspaced;
        }
        double delta = (end_value - start) / (num - 1);
        for (int i = 0; i < num - 1; ++i) {
            linspaced.push_back(start + delta * i);
        }
        linspaced.push_back(end_value);
        return linspaced;
    }

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
    tensorflow::Status Camera_Depth_Estimator::LoadGraph(const tensorflow::string &graph_file_name,
                                                         std::unique_ptr<tensorflow::Session> *session) {
        tensorflow::GraphDef graph_def;
        tensorflow::Status load_graph_status =
                ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
        if (!load_graph_status.ok()) {
            return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                                graph_file_name, "'");
        }
        int node_count = graph_def.node_size();
        for (int i = 0; i < node_count; i++) {
            auto n = graph_def.node(i);
            std::cout << "Names : " << n.name() << std::endl;

        }
        session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
        tensorflow::Status session_create_status = (*session)->Create(graph_def);
        if (!session_create_status.ok()) {
            return session_create_status;
        }
        return tensorflow::Status::OK();
    }

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
    tensorflow::Status
    Camera_Depth_Estimator::GetTopLabels(const std::vector<tensorflow::Tensor> &outputs, int how_many_labels,
                                         tensorflow::Tensor *indices, tensorflow::Tensor *scores) {
        auto root = tensorflow::Scope::NewRootScope();
        using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

        tensorflow::string output_name = "top_k";
        TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
        // This runs the GraphDef network definition that we've just constructed, and
        // returns the results in the output tensors.
        tensorflow::GraphDef graph;
        TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(
                tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_RETURN_IF_ERROR(session->Create(graph));
        // The TopK node returns two outputs, the scores and their original indices,
        // so we have to append :0 and :1 to specify them both.
        std::vector<tensorflow::Tensor> out_tensors;
        TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                        {}, &out_tensors));
        *scores = out_tensors[0];
        *indices = out_tensors[1];
        return tensorflow::Status::OK();
    }

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
    tensorflow::Status Camera_Depth_Estimator::PrintTopLabels(const std::vector<tensorflow::Tensor> &outputs,
                                                              const tensorflow::string &labels_file_name) {
        std::vector<tensorflow::string> labels;
        size_t label_count;
        tensorflow::Status read_labels_status =
                ReadLabelsFile(labels_file_name, &labels, &label_count);
        if (!read_labels_status.ok()) {
            LOG(ERROR) << read_labels_status;
            return read_labels_status;
        }
        const int how_many_labels = std::min(5, static_cast<int>(label_count));
        tensorflow::Tensor indices;
        tensorflow::Tensor scores;
        TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
        tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
        tensorflow::TTypes<tensorflow::int32>::Flat indices_flat = indices.flat<tensorflow::int32>();
        for (int pos = 0; pos < how_many_labels; ++pos) {
            const int label_index = indices_flat(pos);
            const float score = scores_flat(pos);
            LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
        }
        return tensorflow::Status::OK();
    }

// This is a testing function that returns whether the top label index is the
// one that's expected.
    tensorflow::Status
    Camera_Depth_Estimator::CheckTopLabel(const std::vector<tensorflow::Tensor> &outputs, int expected,
                                          bool *is_expected) {
        *is_expected = false;
        tensorflow::Tensor indices;
        tensorflow::Tensor scores;
        const int how_many_labels = 1;
        TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
        tensorflow::TTypes<tensorflow::int32>::Flat indices_flat = indices.flat<tensorflow::int32>();
        if (indices_flat(0) != expected) {
            LOG(ERROR) << "Expected label #" << expected << " but got #"
                       << indices_flat(0);
            *is_expected = false;
        } else {
            *is_expected = true;
        }
        return tensorflow::Status::OK();
    }

    void Camera_Depth_Estimator::getDisparityMap(cv::Mat current_image,  cv::Mat &disparity_map) {
        try {
            cv::Mat flip_current_image;
            cv::flip(current_image, flip_current_image, 1);
            //cv::imshow("view",current_image);
            next_image += 1;
            //std::tensorflow::string next_img = "_input_image.png";
            //cv::imwrite(std::to_tensorflow::string(next_image) + next_img,  current_image);

            int depth = current_image.channels();
            cv::resize(current_image, current_image, cv::Size(input_height, input_width), 0, 0);

            tensorflow::Tensor input_tensor(
                    tensorflow::DT_FLOAT,
                    tensorflow::TensorShape({2, input_height, input_width, depth}));

            auto input_tensor_mapped = input_tensor.tensor<float, 4>();
            LOG(INFO) << "TensorFlow: Copying Data.";

//        const float* source_data = (float*) current_image.data;
            float max_value = 0;
            float min_value = 0;
            for (int y = 0; y < input_height; ++y) {
//            float* source_row = current_image.ptr(y);
                for (int x = 0; x < input_width; ++x) {
                    for (int c = depth - 1; c >= 0; --c) {
                        int current_value = current_image.at<cv::Vec3b>(y, x)[c] / 255;
                        if (min_value > current_value) {
                            min_value = current_value;
                        }
                        if (max_value < current_value) {
                            max_value = current_value;
                        }
                        input_tensor_mapped(0, y, x, c) = current_value;
                        input_tensor_mapped(1, y, x, c) = flip_current_image.at<cv::Vec3b>(y, x)[c] / 255;
                    }
                    /*float* source_pixel = source_row + (x * depth);
                    for (int c = depth-1; c>=0; --c) {
                        const float* source_value = source_pixel + c;
                        input_tensor_mapped(0, y, x, c) = *source_value/255;
    //                  std::cout<< *source_value/255 << std::endl;
                    }*/
                }
            }
            std::cout << "max value: " << max_value << "min value: " << min_value << std::endl;
            const tensorflow::Tensor &resized_tensor = input_tensor;

            // Actually run the image through the model.
            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status run_status = session->Run({{input_layer, resized_tensor}},
                                                         {output_layer}, {}, &outputs);
            if (!run_status.ok()) {
                LOG(ERROR) << "Running model failed: " << run_status;
                return;
            } else {
                std::cout << "Model loading is successful: " << outputs.at(0).DebugString() << std::endl;
            }
            auto finalOutputTensor1 = outputs.at(0).tensor<float, 4>();
            cv::Mat disparity_map_left = cv::Mat::zeros(cv::Size(input_height, input_width), CV_32FC1);
            cv::Mat disparity_map_right = cv::Mat::zeros(cv::Size(input_height, input_width), CV_32FC1);

            std::vector<double> steps = linspace(0, 1, input_width);
            cv::Mat mask_left = cv::Mat::zeros(cv::Size(input_height, input_width), CV_32FC1);

            for (int i = 0; i < input_height; i++) {
                for (int j = 0; j < input_width; j++) {
                    disparity_map_left.at<double>(i, j) = finalOutputTensor1(0, i, j, 0);
                    disparity_map_right.at<double>(i, j) = finalOutputTensor1(1, i, j, 0);
                    mask_left.at<double>(i, j) = steps.at(j);
                }
            }
            cv::Mat disparity_map_average = (disparity_map_left + disparity_map_right) / 2;
            mask_left = 20 * (mask_left - 0.05);
            for (int i = 0; i < input_height; i++) {
                for (int j = 0; j < input_width; j++) {
                    if (mask_left.at<double>(i, j) > 1) {
                        mask_left.at<double>(i, j) = 1.0;
                    }
                    if (mask_left.at<double>(i, j) < 0) {
                        mask_left.at<double>(i, j) = 0.0;
                    }
                }
            }
            mask_left = 1.0 - mask_left;
            cv::Mat mask_right;
            cv::flip(mask_left, mask_right, 1);

            cv::Mat depth_map = mask_right.mul(disparity_map_left) + mask_left.mul(disparity_map_right)
                                + (1 - mask_left - mask_right).mul(disparity_map_average);

            std::ofstream outputfile;
            std::string next_map = std::to_string(next_image) + "_disparity_map.csv";
            outputfile.open(next_map);

            for (int count = 0; count < input_height; count++) {
                for (int index = 0; index < input_width; index++)
                    outputfile << depth_map.at<double>(count,
                                                       index);//outputfile << finalOutputTensor1(0, count, index, 0)<<",";
                outputfile << std::endl;
            }
            outputfile.close();

            disparity_map = depth_map;


//        std::string next_map = std::to_string(next_image) + "_disparity_map.exr";
//        cv::FileStorage file( next_map.c_str(), cv::FileStorage::WRITE);
//        file<< "disparity_map" << disparity_map;

            //        cv::imwrite(std::to_string(next_image) + next_map,  disparity_map);

            //        cv::Size s = disparity_map.size();
            //        int rows = s.height;
            //        int cols = s.width;
            //        std::cout<< "Size, if the image can be define as bellow:" << rows <<  cols <<std::endl;
            // std::cout<<outputs.at(0).DebugString() << std::endl;

            //     *     if (self_test) {
            //        bool expected_matches;
            //        tensorflow::Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
            //        if (!check_status.ok()) {
            //            LOG(ERROR) << "Running check failed: " << check_status;
            //            return -1;
            //        }
            //        if (!expected_matches) {
            //            LOG(ERROR) << "Self-test failed!";
            //            return -1;
            //        }
            //    }
            //
            //    // Do something interesting with the results we've generated.
            //    tensorflow::Status print_status = PrintTopLabels(outputs, labels);
            //    if (!print_status.ok()) {
            //        LOG(ERROR) << "Running print failed: " << print_status;
            //        return -1;
            //    }
            //cv::waitKey(30);
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("Could not convert from error while calculating depth map.");
        }
    }

//    static void wrapper_imageCallback(void *pt2Object, const sensor_msgs::ImageConstPtr &msg) {
//        Camera_Depth_Estimator *caller_class = (Camera_Depth_Estimator *) pt2Object;
//        caller_class->imageCallback(msg);
//    }

}
/*
int main(int argc, char* argv[]) {

    Camera_Depth_Estimator camera_depth_estimator;
    string usage = tensorflow::Flags::Usage(argv[0], camera_depth_estimator.flag_list);
    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, camera_depth_estimator.flag_list);
    typedef const boost::function< void(const sensor_msgs::ImageConstPtr &)>  callback;

    if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
    }

    tensorflow::port::InitMain(argv[0], &argc, &argv);
    if (argc > 1) {
        LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
        return -1;
    }

    camera_depth_estimator.session = std::unique_ptr<tensorflow::Session>();

    string graph_path = tensorflow::io::JoinPath(camera_depth_estimator.root_dir, camera_depth_estimator.graph);
    tensorflow::Status load_graph_status = camera_depth_estimator.LoadGraph(graph_path, &camera_depth_estimator.session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    }else{
        std::cout<<"Graph loading is successful"<<std::endl;
    }

    ros::init(argc, argv, "image_subscriber_depth_estimator");
    ros::NodeHandle nh;
    ros::Subscriber sub_front;
    callback boundImageCallback = boost::bind(&Camera_Depth_Estimator::imageCallback, &camera_depth_estimator, _1);
    sub_front = nh.subscribe(camera_depth_estimator.front_camera, 1, boundImageCallback);
    ros::spin();
    ROS_ERROR("Exit");
    return 0;
}*/
