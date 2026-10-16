// clahe_bench.cpp: measures how long CLAHE takes per image, so it can be added
// to the front-end time when judging real-time performance.
//
// Build:
//   g++ -O2 -std=c++14 clahe_bench.cpp -o clahe_bench $(pkg-config --cflags --libs opencv4)
//
// Run (use the same clip limit, tile grid and colour handling as your offline preprocessing):
//   ./clahe_bench <image_folder> [clip_limit=2.0] [tile_grid=8] [mode=gray|lab] [max_images=300]
//
//   mode=gray  CLAHE on the grayscale image (ORB-SLAM3 only uses grayscale)
//   mode=lab   CLAHE on the L channel of LAB, converted back to BGR; the colour
//              conversions are included in the time
//
// Images are loaded into memory before timing, so disk reads are not counted.
// Run it on the same machine, under the same conditions, as your SLAM timing runs.

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <image_folder> [clip_limit=2.0] [tile_grid=8] [mode=gray|lab] [max_images=300]" << std::endl;
        return 1;
    }

    const std::string folder = argv[1];
    const double clipLimit = argc > 2 ? std::atof(argv[2]) : 2.0;
    const int tileGrid = argc > 3 ? std::atoi(argv[3]) : 8;
    const std::string mode = argc > 4 ? argv[4] : "gray";
    const size_t maxImages = argc > 5 ? static_cast<size_t>(std::atoi(argv[5])) : 300;

    if (mode != "gray" && mode != "lab")
    {
        std::cerr << "mode must be gray or lab" << std::endl;
        return 1;
    }

    std::vector<cv::String> files;
    try
    {
        cv::glob(folder + "/*.png", files, false);
        std::vector<cv::String> jpgs;
        cv::glob(folder + "/*.jpg", jpgs, false);
        files.insert(files.end(), jpgs.begin(), jpgs.end());
    }
    catch (const cv::Exception&)
    {
        std::cerr << "Cannot open folder " << folder << std::endl;
        return 1;
    }
    std::sort(files.begin(), files.end());

    std::vector<cv::Mat> images;
    for (size_t i = 0; i < files.size() && images.size() < maxImages; i++)
    {
        cv::Mat im = cv::imread(files[i], mode == "gray" ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
        if (!im.empty())
            images.push_back(im);
    }
    if (images.empty())
    {
        std::cerr << "No .png or .jpg images found in " << folder << std::endl;
        return 1;
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileGrid, tileGrid));
    cv::Mat out, lab;
    std::vector<cv::Mat> channels;

    auto runOnce = [&](const cv::Mat& im) {
        if (mode == "gray")
        {
            clahe->apply(im, out);
        }
        else
        {
            cv::cvtColor(im, lab, cv::COLOR_BGR2Lab);
            cv::split(lab, channels);
            clahe->apply(channels[0], channels[0]);
            cv::merge(channels, lab);
            cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
        }
    };

    // Warm-up: the first calls allocate buffers and would inflate the mean
    for (size_t i = 0; i < std::min<size_t>(5, images.size()); i++)
        runOnce(images[i]);

    std::vector<double> times;
    times.reserve(images.size());
    for (size_t i = 0; i < images.size(); i++)
    {
        const auto t0 = std::chrono::steady_clock::now();
        runOnce(images[i]);
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double sum = 0.0;
    for (double t : times)
        sum += t;
    const double mean = sum / times.size();
    double sq = 0.0;
    for (double t : times)
        sq += (t - mean) * (t - mean);
    const double stdDev = times.size() > 1 ? std::sqrt(sq / (times.size() - 1)) : 0.0;
    std::sort(times.begin(), times.end());
    const double median = (times.size() % 2 == 1)
                              ? times[times.size() / 2]
                              : 0.5 * (times[times.size() / 2 - 1] + times[times.size() / 2]);

    std::printf("Images: %zu  (%dx%d, mode=%s, clip=%.2f, tiles=%dx%d)\n", times.size(), images[0].cols,
                images[0].rows, mode.c_str(), clipLimit, tileGrid, tileGrid);
    std::printf("CLAHE per image   (ms): mean %.2f  median %.2f  std %.2f\n", mean, median, stdDev);
    std::printf("CLAHE per stereo pair, left + right (ms): mean %.2f\n", 2.0 * mean);
    return 0;
}
