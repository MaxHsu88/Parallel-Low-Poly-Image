#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

int main() {
    string image_path = "./img/patrick.jpg";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    cv::imwrite("starry_night.jpg", img);

    return 0;
}