#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"


#define OUTPUT_FILE "../../../submission.csv"
#define IMAGE_DIRECTORY "../../../data/images"


int main(int argc, char* argv[])
{

    //************************COMMAND LINE SETUP************************//
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;
    std::ofstream file(OUTPUT_FILE);
    std::vector<std::__fs::filesystem::path> imagePaths;

    
    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", false, "../models/yolov5s.onnx");
    cmd.add<std::string>("image_directory", 'i', "Image directory.", false, IMAGE_DIRECTORY);
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "../models/coco.names");
    cmd.add("gpu", '\0', "Inference on cuda device.");
    
    

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string imageDirectoryPath = cmd.get<std::string>("image_directory");
    const std::string modelPath = cmd.get<std::string>("model_path");
    

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }


    //Image batches
    utils::createImageBatch(imageDirectoryPath, imagePaths);

    //************************DETECTOR SETUP************************//
    YOLODetector detector {nullptr};
    cv::Mat image;
    std::vector<Detection> result;

    std::cout << "Starting inference..." << std::endl;
    std::cout << "Number of images: " << imagePaths.size() << std::endl;


    for (const auto& imagePath : imagePaths) {
        try
        {
            detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
            std::cout << "Model was initialized." << std::endl;

            image = cv::imread(imagePath.string());

            //A vector of detection objects
            result = detector.detect(image, confThreshold, iouThreshold);
            utils::writeToCSV(result, imagePath.filename().string(), file);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            return -1;
        }

        /*
        //Visualizing detection
        utils::visualizeDetection(image, result, classNames);

        //Opens the window
        cv::imshow("result", image);
        // cv::imwrite("result.jpg", image);
        cv::waitKey(0);
        */
        
    }

    return 0;
}
