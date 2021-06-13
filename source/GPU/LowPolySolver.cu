 #include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "point.h"
#include "triangle.h"
#include "delauney.h"
#include "img_utils.h"

using namespace std;


int main(int argc, char **argv)
{
    string image_path = argv[1];
    string output_folder = argv[2];
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

    select_vertices_GPU(grey_img, gradient_img, height, width);

    // ****************************
    // for output edge image
    cv::Mat edge_output = drawEdges(gradient_img, height, width);
    cv::imwrite( output_folder + "/GPU_edge.jpg", edge_output );
    // ****************************

    // float gradThreshold = 20;
    // float edgeProb = 0.005;
    // float nonEdgeProb = 0.0001;
    // float boundProb = 0.1;
    // vector<Point> vertices = selectVertices(gradient_img, height, width, gradThreshold, edgeProb, nonEdgeProb, boundProb);

    // // ****************************
    // // for output vertex image
    // // cv::Mat vertex_output = drawVert(vertices, height, width);
    // // cv::imwrite( output_folder + "/vertex.jpg", vertex_output );
    // // ****************************

    // vector<int> owner(height * width, -1);
    // vector<Triangle> triangles = Delauney(vertices, owner, height, width);

    // // ****************************
    // // for output voroni images
    // int num_vertices = vertices.size();
    // cv::Mat voroni_output = drawVoroni(owner, num_vertices, height, width);
    // cv::imwrite( output_folder + "/voroni.jpg", voroni_output );
    // // for output voroni + triangle images
    // cv::Mat voroni_tri_output = drawTriangles(triangles, voroni_output, true);
    // cv::imwrite( output_folder + "/voroni_triangle.jpg", voroni_tri_output );
    // // for output triangle on original images
    // cv::Mat orig_tri_output = drawTriangles(triangles, img, true);
    // cv::imwrite( output_folder + "/original_triangle.jpg", orig_tri_output );
    // // ****************************

    // // ****************************
    // // for output final image
    // cv::Mat final_output = drawLowPoly(triangles, img, height, width);
    // cv::imwrite( output_folder + "/final.jpg", final_output );
    // ****************************

    free(gradient_img);

    return 0;
}