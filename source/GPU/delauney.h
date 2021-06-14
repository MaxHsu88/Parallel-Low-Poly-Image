// Avoid repeated macro defined
#pragma once

#include <vector>
#include <unordered_set>

#include "cuda.h"

#include "point.h"
#include "triangle.h"
#include "delauney.h"

using namespace std;

void delauney_GPU(Point *owner_map_CPU, vector<Triangle> &triangles_list, int height, int width);
void select_vertices_GPU(uint8_t *grey_img_CPU, uint8_t *result_image, int height, int width);
cv::Mat drawLowPoly_GPU(cv::Mat &img);
void free_gpu_memory();