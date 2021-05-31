// Avoid repeated macro defined
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "point.h"
#include "triangle.h"

using namespace std;

cv::Mat drawTriangles(vector<Triangle> &triangles, cv::Mat &orig_img, int height, int width);