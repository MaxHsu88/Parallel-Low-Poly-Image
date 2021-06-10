// Avoid repeated macro defined
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "point.h"
#include "triangle.h"

using namespace std;

// Image processing unit
void get_gradient(uint8_t *grey_img, uint8_t *gradient_img, int height, int width);
vector<Point> selectVertices(uint8_t *grad, int height, int width, float gradThreshold, float edgeProb, float nonEdgeProb, float boundProb);

// Image drawing unit
cv::Mat drawLowPoly(vector<Triangle> &triangles, cv::Mat &orig_img, int height, int width);
cv::Mat drawTriangles(vector<Triangle> &triangles, cv::Mat &img, bool add);
cv::Mat drawEdges(uint8_t* gradient_img, int height, int width);
cv::Mat drawVert(vector<Point> &vertices, int height, int width);
cv::Mat drawVoroni(vector<int> &owner, int num_vertices, int height, int width);