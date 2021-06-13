#include <vector>
#include <unordered_set>

#include "cuda.h"

#include "point.h"
#include "triangle.h"
#include "delauney.h"

#define MASK_N 2
#define MASK_X 3
#define MASK_Y 3
#define SCALE 8

__constant__ int adjustX = (MASK_X % 2) ? 1 : 0;
__constant__ int adjustY = (MASK_Y % 2) ? 1 : 0;
__constant__ int xBound = MASK_X / 2;
__constant__ int yBound = MASK_Y / 2;

uint8_t *grey_img_GPU;
uint8_t *gradient_img_GPU;
uint8_t *owner_map_GPU;


inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    }
    return result;
}


// Most of this code is borrowed from Homework3
__global__
void get_gradient_kernel(uint8_t *grey_img, uint8_t *gradient_img, int height, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int mask[MASK_N][MASK_X][MASK_Y] = {
        {{1, 0, -1},
         {2, 0, -2},
         {1, 0, -1}},
        {{1, 2, 1},
         {0, 0, 0},
         {-1, -2, -1}}
    };

    if (y >= 0 && y < height && x >= 0 && x < width) {
        float grad[2] = {0.0, 0.0};   // grad-x and grad-y

        for (int i = 0; i < MASK_N; ++i) {
            grad[i] = 0.0;
            for (int v = -yBound; v < yBound + adjustY; ++v) {
                for (int u = -xBound; u < xBound + adjustX; ++u) {
                    if ((x + u) >= 0 && (x + u) < width && (y + v) >= 0 && (y + v) < height) {
                        grad[i] += grey_img[width * (y + v) + (x + u)] * mask[i][u + xBound][v + yBound];
                    }
                }
            }
            grad[i] = abs(grad[i]);
        }

        float total_grad = grad[0] / 2.0 + grad[1] / 2.0;
        const unsigned char c = (total_grad > 255.0) ? 255 : total_grad;
        gradient_img[y * width + x] = c;
    }
}


vector<Point> select_vertices_kernel(uint8_t *grad, int height, int width, float gradThreshold, float edgeProb, float nonEdgeProb, float boundProb)
{
    vector<Point> vertices;
    uint8_t gradVal;

    // four corners must be in the set
    Point p1(0, 0);
    Point p2(0, height-1);
    Point p3(width-1, 0);
    Point p4(width-1, height-1);

    vertices.push_back(p1);
    vertices.push_back(p2);
    vertices.push_back(p3);
    vertices.push_back(p4);

    // boundary area conditions
    for (int row = 1; row < height-1; row++)
    {
        // left-most boundary
        double randNum = (double) rand() / RAND_MAX;
        if (randNum < boundProb)
        {
            Point p(0, row);
            vertices.push_back(p);
        }
        // right-most boundary
        randNum = (double) rand() / RAND_MAX;
        if (randNum < boundProb)
        {
            Point p(width-1, row);
            vertices.push_back(p);
        }
    }
    for (int col = 1; col < width-1; col++)
    {
        // up-most boundary
        double randNum = (double) rand() / RAND_MAX;
        if (randNum < boundProb)
        {
            Point p(col, 0);
            vertices.push_back(p);
        }
        // down-most boundary
        randNum = (double) rand() / RAND_MAX;
        if (randNum < boundProb)
        {
            Point p(col, height-1);
            vertices.push_back(p);
        }
    }

    // inner area conditions
    for (int i = 1; i < height-1; i++)
    {
        for (int j = 1; j < width-1; j++)
        {
            gradVal = grad[i * width + j];
            double randNum = (double) rand() / RAND_MAX;
            if (gradVal > gradThreshold)
            {
                // Edge vertex
                if (randNum < edgeProb)
                {
                    Point p(j, i);
                    vertices.push_back(p);
                }
            }
            else
            {
                // Non-edge vertex
                if (randNum < nonEdgeProb)
                {
                    Point p(j, i);
                    vertices.push_back(p);
                }
            }
        }
    }

    return vertices;
}


void select_vertices_GPU(uint8_t *grey_img_CPU, uint8_t *result_img, int height, int width)
{
    int total_pixels = height * width;

    // GPU memory allocation
    cudaMalloc(&grey_img_GPU, total_pixels * sizeof(uint8_t));
    cudaMalloc(&gradient_img_GPU, total_pixels * sizeof(uint8_t));
    cudaMalloc(&owner_map_GPU, total_pixels * sizeof(int));

    // Data transfer
    cudaMemcpy(grey_img_GPU, grey_img_CPU, total_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Edge detection filtering
    int GRIDSIZE = 32;
    dim3 dimBlock(GRIDSIZE, GRIDSIZE, 1);
    dim3 dimGrid(ceil(width/(float)GRIDSIZE), ceil(height/(float)GRIDSIZE), 1);
    get_gradient_kernel<<<dimGrid, dimBlock>>>(grey_img_GPU, gradient_img_GPU, height, width);

    // get the result back
    cudaMemcpy(result_img, gradient_img_GPU, total_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Selecting vertices

    // Free memory
    cudaFree(grey_img_GPU);
    cudaFree(gradient_img_GPU);
    cudaFree(owner_map_GPU);
}


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

    return triangles;
}