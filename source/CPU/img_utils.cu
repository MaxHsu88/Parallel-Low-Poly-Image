#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "point.h"
#include "triangle.h"

using namespace std;

#define MASK_N 2
#define MASK_X 3
#define MASK_Y 3

#define SCALE 8

// Most of this code is borrowed from Homework3
void get_gradient(uint8_t *grey_img, uint8_t *gradient_img, int height, int width)
{
    int mask[MASK_N][MASK_X][MASK_Y] = {
        {{1, 0, -1},
         {2, 0, -2},
         {1, 0, -1}},
        {{1, 2, 1},
         {0, 0, 0},
         {-1, -2, -1}}
    };

    int x, y, u, v, i;
    int adjustX, adjustY, xBound, yBound;
    adjustX = (MASK_X % 2) ? 1 : 0;
    adjustY = (MASK_Y % 2) ? 1 : 0;
    xBound = MASK_X / 2;
    yBound = MASK_Y / 2;

    float grad[2] = {0.0, 0.0};   // grad-x and grad-y

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (i = 0; i < MASK_N; ++i) {
                grad[i] = 0.0;
                for (v = -yBound; v < yBound + adjustY; ++v) {
                    for (u = -xBound; u < xBound + adjustX; ++u) {
                        if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
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
}


vector<Point> selectVertices(uint8_t *grad, int height, int width, float gradThreshold, float edgeProb, float nonEdgeProb, float boundProb)
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

// **************************************
// This code checks if a point (pt) lies in a triangle (v1, v2, v3)
// Reference: https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
float sign(Point p1, Point p2, Point p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool PointInTriangle(Point pt, Point v1, Point v2, Point v3)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}
// **************************************


// Draw the final triangulation images (with color)
cv::Mat drawLowPoly(vector<Triangle> &triangles, cv::Mat &orig_img, int height, int width)
{
    cv::Mat out_img = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    // Iterate for all triangles
    for (int i = 0; i < triangles.size(); i++)
    {
        Triangle triangle = triangles[i];
        // Use center pixel of a triangle to color it
        Point pt_c = triangle.get_center();
        // Find bounding box region of a triangle
        int minX = min(triangle.points[0].x, min(triangle.points[1].x, triangle.points[2].x));
        int maxX = max(triangle.points[0].x, max(triangle.points[1].x, triangle.points[2].x));
        int minY = min(triangle.points[0].y, min(triangle.points[1].y, triangle.points[2].y));
        int maxY = max(triangle.points[0].y, max(triangle.points[1].y, triangle.points[2].y));
        // Iterate for the pixels in the box region
        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                Point pt_tmp(x, y);
                // Check if the pixels lies in the triangle
                if (PointInTriangle(pt_tmp, triangle.points[0], triangle.points[1], triangle.points[2]))
                {
                    // Assign the color of ceter pixel of the triangle to current pixel
                    out_img.at<cv::Vec3b>(y, x) = orig_img.at<cv::Vec3b>(pt_c.y, pt_c.x);
                }
            }
        }
    }

    return out_img;
}


// Draw the edge detection images
cv::Mat drawEdges(uint8_t* gradient_img, int height, int width)
{
    cv::Mat edge_output = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            edge_output.at<uchar>(i, j) = gradient_img[i * width + j];
        }
    }
    return edge_output;
}


// Draw the selected vertex images
cv::Mat drawVert(vector<Point> &vertices, int height, int width)
{
    cv::Mat vertex_output = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < vertices.size(); ++i)
    {
        Point p = vertices[i];
        int x = p.x;
        int y = p.y;
        vertex_output.at<uchar>(y, x) = 255;
    }
    return vertex_output;
}


// Draw the voroni images
cv::Mat drawVoroni(vector<int> &owner, int num_vertices, int height, int width)
{
    cv::Mat voroni_output = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // Randomized RGB color for each region
    vector<cv::Vec3b> vertices_color(num_vertices);
    for (int i = 0; i < num_vertices; i++)
    {
        vertices_color[i][0] = rand() % 256;
        vertices_color[i][1] = rand() % 256;
        vertices_color[i][2] = rand() % 256;
    }

    // Assign each pixel with corresponding region color
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            voroni_output.at<cv::Vec3b>(i, j) = vertices_color[owner[i * width + j]];
        }
    }

    return voroni_output;
}


// Draw the triangulation results
cv::Mat drawTriangles(vector<Triangle> &triangles, cv::Mat &img, bool add)
{
    cv::Mat triangle_output;
    // there are two different operations
    // add is True : overwrite the input image with triangulation results
    // add is False: only write triangulation results in a new image
    if (add)
    {
        triangle_output = img.clone();
        for (int i = 0; i < triangles.size(); i++)
        {
            Triangle tri = triangles[i];
            cv::Point p1, p2, p3;
            p1.x = tri.points[0].x;
            p1.y = tri.points[0].y;
            p2.x = tri.points[1].x;
            p2.y = tri.points[1].y;
            p3.x = tri.points[2].x;
            p3.y = tri.points[2].y;
            cv::line(triangle_output, p1, p2, cv::Scalar( 0, 0, 0));
            cv::line(triangle_output, p2, p3, cv::Scalar( 0, 0, 0));
            cv::line(triangle_output, p3, p1, cv::Scalar( 0, 0, 0));
        }
    }
    else
    {
        int height = img.rows;  int width = img.cols;
        triangle_output = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < triangles.size(); i++)
        {
            Triangle tri = triangles[i];
            cv::Point p1, p2, p3;
            p1.x = tri.points[0].x;
            p1.y = tri.points[0].y;
            p2.x = tri.points[1].x;
            p2.y = tri.points[1].y;
            p3.x = tri.points[2].x;
            p3.y = tri.points[2].y;
            cv::line(triangle_output, p1, p2, cv::Scalar( 255, 255, 255));
            cv::line(triangle_output, p2, p3, cv::Scalar( 255, 255, 255));
            cv::line(triangle_output, p3, p1, cv::Scalar( 255, 255, 255));
        }
    }
    return triangle_output;
}