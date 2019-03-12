#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

constexpr int H_RED = 2;
constexpr int H_GREEN = 1;
constexpr int H_BLUE = 0;

class Histogram:cv::Mat
{
	int minV, maxV;
	int nob;

public:


	Histogram(const cv::Mat& m, int maxi = 255, int mini = -255);

	~Histogram();

	void normalize(float min, float max);

	void draw(const char* winName);

	float get(int channel, int value) const;

	void set(int channel, int value,float newVal) ;
	cv::Mat getChannel(int channel) const;

	void smooth(int k);
};
