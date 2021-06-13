// Avoid repeated macro defined
#pragma once

#include <vector>
#include <unordered_set>

#include "cuda.h"

#include "point.h"
#include "triangle.h"
#include "delauney.h"

using namespace std;

vector<Triangle> Delauney(vector<Point> &vertices, vector<int> &owner, int height, int width);

void select_vertices_GPU(uint8_t *grey_img_CPU, uint8_t *result_img, int height, int width);