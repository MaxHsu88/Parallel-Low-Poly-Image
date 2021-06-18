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
#include "simpleTimer.h"

using namespace std;


int main(int argc, char **argv)
{
    simpleTimer t_whole("whole process");

    // init_cuda();

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

    simpleTimer t_mem_alloc("GPU memory setup");

    setup_gpu_memory(height, width);

    t_mem_alloc.GetDuration();

    simpleTimer t_edge("Edge detection & select vertices");

    cv::Mat img_grey;
    cv::cvtColor(img, img_grey, cv::COLOR_BGR2GRAY);

    int totalPixel = height * width;
    uint8_t *gradient_img = (uint8_t *)malloc(sizeof(uint8_t) * totalPixel);
    uint8_t *grey_img = img_grey.data;

    Point *vert_img = (Point *)malloc(sizeof(Point) * totalPixel);

    select_vertices_GPU(grey_img, gradient_img, vert_img, height, width);

    t_edge.GetDuration();

    // ****************************
    // for output edge image
    // cv::Mat edge_output = drawEdges(gradient_img, height, width);
    // cv::imwrite( output_folder + "/GPU_edge.jpg", edge_output );
    // ****************************

    // ****************************
    // for output vertices image
    // cv::Mat vertices_output = drawVert(vert_img, height, width);
    // cv::imwrite( output_folder + "/GPU_vertex.jpg", vertices_output );
    // ****************************

    simpleTimer t_delauney("Delauney Triangulation");

    Point *owner_map_CPU = (Point *)malloc(sizeof(Point) * totalPixel);
    vector<Triangle> triangles;
    delauney_GPU(owner_map_CPU, triangles, height, width);

    t_delauney.GetDuration();

    // ****************************
    // for output voroni images
    // cv::Mat voroni_output = drawVoroni(owner_map_CPU, height, width);
    // cv::imwrite( output_folder + "/GPU_voroni.jpg", voroni_output );
    // // for output voroni + triangle images
    // cv::Mat voroni_tri_output = drawTriangles(triangles, voroni_output, true);
    // cv::imwrite( output_folder + "/GPU_voroni_triangle.jpg", voroni_tri_output );
    // // for output triangle on original images
    // cv::Mat orig_tri_output = drawTriangles(triangles, img, true);
    // cv::imwrite( output_folder + "/GPU_original_triangle.jpg", orig_tri_output );
    // ****************************

    simpleTimer t_color("Coloring Image");

    // ****************************
    // for output final image
    cv::Mat final_output = drawLowPoly_GPU(img);
    cv::imwrite( output_folder + "/GPU_final.jpg", final_output );
    // ****************************

    t_color.GetDuration();

    free(gradient_img);
    free(owner_map_CPU);
    free(vert_img);
    free_gpu_memory();

    t_whole.GetDuration();

    return 0;
}