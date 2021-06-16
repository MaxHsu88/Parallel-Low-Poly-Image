#include <vector>
#include <unordered_set>

#include "point.h"
#include "triangle.h"
#include "delauney.h"

#include "simpleTimer.h"


// Get the ceiling of the value which is power of 2
// Reference: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
int ceil_power2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

inline int convert_idx(Point p, int width)
{
    return p.y * width + p.x;
}

inline bool out_of_bound(Point p, int height, int width)
{
    return !(p.x >= 0 && p.x < width && p.y >= 0 && p.y < height);
}



vector<Triangle> Delauney(vector<Point> &vertices, vector<int> &owner, int height, int width)
{
    simpleTimer t_jump_flood("...Jump flooding");

    // All 8 directions to check from the vertex
    const Point all_dir[8] = {Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
                                Point(-1, 0), Point(-1, -1), Point(0, -1), Point(1, -1)};

    // Assign each sampled vertex's index as its located pixel's owner
    for (int i = 0; i < vertices.size(); i++)
    {
        Point vertex = vertices[i];
        owner[convert_idx(vertex, width)] = i;
    }

    // **************************************
    // Jump-Flooding algorithm for constructing voronoi diagram
    // Reference: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.101.8568&rep=rep1&type=pdf
    // **************************************

    int init_step_size = ceil_power2(min(height, width)) / 2;
    // Iterate possible step sizes
    for (int step_size = init_step_size; step_size >= 1; step_size /= 2)
    {
        // Check for all the pixels
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Point cur_point(x, y);
                // Check for all possible directions to neighbor points
                for (int i = 0; i < 8; i++)
                {
                    Point cur_dir = all_dir[i];
                    Point cur_looking = cur_point + cur_dir * step_size;
                    // If this point is out of bounds, skip it
                    if (out_of_bound(cur_looking, height, width))
                    {
                        continue;
                    }
                    // If this point is not owned by anyone, skip it
                    if (owner[convert_idx(cur_looking, width)] == -1)
                    {
                        continue;
                    }

                    // Update owner in cur_point only when
                    // 1. cur_point is NOT owned by anyone (owner = -1)
                    // 2. cur_point has shorter distance to cur_looking's owner than previous owner
                    int cur_owner = owner[convert_idx(cur_point, width)];
                    int tmp_dist = distance(vertices[owner[convert_idx(cur_looking, width)]], cur_point);
                    if (cur_owner == -1 || tmp_dist < distance(vertices[cur_owner], cur_point))
                    {
                        owner[convert_idx(cur_point, width)] = owner[convert_idx(cur_looking, width)];
                    }
                }
            }
        }
    }

    t_jump_flood.GetDuration();

    simpleTimer t_build_tri("...Building triangles");

    // **************************************
    // Building triangles from the voronoi diagram
    // **************************************
    vector<Triangle> triangles;
    const Point corner_dir[3] = {Point(0, 1), Point(1, 0), Point(1, 1)};
    // Check for all the pixels
    for (int y = 0; y < height - 1; y++)
    {
        for (int x = 0; x < width - 1; x++)
        {
            // Push the owners of itself and its neighboring point to the set
            unordered_set<int> owner_set;
            Point cur_point(x, y);
            owner_set.insert(owner[convert_idx(cur_point, width)]);
            for (int i = 0; i < 3; i++)
            {
                Point neighbor_point = cur_point + corner_dir[i];
                owner_set.insert(owner[convert_idx(neighbor_point, width)]);
            }

            // If 3 distinct owners in the corner, there exists 1 triangle
            if (owner_set.size() == 3)
            {
                Triangle triangle;
                int k = 0;
                for (const auto &p: owner_set)
                {
                    triangle.points[k] = vertices[p];
                    k++;
                }
                triangles.push_back(triangle);
            }

            // If 4 distinct owners in the corner, there exists 2 triangles
            if (owner_set.size() == 4)
            {
                Triangle triangle1, triangle2;
                triangle1 = Triangle(vertices[owner[convert_idx(cur_point, width)]],
                                    vertices[owner[convert_idx(cur_point + corner_dir[0], width)]],
                                    vertices[owner[convert_idx(cur_point + corner_dir[1], width)]]);
                
                triangle2 = Triangle(vertices[owner[convert_idx(cur_point + corner_dir[0], width)]],
                                    vertices[owner[convert_idx(cur_point + corner_dir[1], width)]],
                                    vertices[owner[convert_idx(cur_point + corner_dir[2], width)]]);

                triangles.push_back(triangle1);
                triangles.push_back(triangle2);
            }
        }
    }

    t_build_tri.GetDuration();

    return triangles;
}