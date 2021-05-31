#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "point.h"

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


vector<Point> selectVertices(uint8_t *grad, int height, int width)
{
    vector<Point> vertices;
    uint8_t gradVal;

    // define parameters for vertex selection
    float gradThreshold = 20;
    float edgeProb = 0.3;
    float nonEdgeProb = 0.01;
    float boundProb = 0.2;

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


int main()
{
    string image_path = "../img/patrick.jpg";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
 
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    int height = img.rows;
    int width = img.cols;

    cout << "height: " << height << ", width: " << width << endl;

    cv::Mat img_grey;
    cv::cvtColor(img, img_grey, cv::COLOR_BGR2GRAY);

    int totalPixel = height * width;
    uint8_t *gradient_img = (uint8_t *)malloc(sizeof(uint8_t) * height * width);
    uint8_t *grey_img = img_grey.data;

    get_gradient(grey_img, gradient_img, height, width);

    vector<Point> vertices = selectVertices(gradient_img, height, width);


    // for output edge image
    // cv::Mat edge_output = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    // for (int i = 0; i < height; ++i)
    // {
    //     for (int j = 0; j < width; ++j)
    //     {
    //         edge_output.at<uchar>(i, j) = gradient_img[i * width + j];
    //     }
    // }

    // for output vertex image
    cv::Mat vertex_output = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < vertices.size(); ++i)
    {
        Point p = vertices[i];
        int x = p.x;
        int y = p.y;
        vertex_output.at<uchar>(y, x) = 255;
    }

    cv::imwrite("edge.jpg", vertex_output);

    free(gradient_img);

    return 0;
}