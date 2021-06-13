// Avoid repeated macro defined
#pragma once

#include <vector>
#include <unordered_set>

#include "point.h"
#include "triangle.h"
#include "delauney.h"

using namespace std;

vector<Triangle> Delauney(vector<Point> &vertices, vector<int> &owner, int height, int width);