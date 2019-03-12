#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>

#include "Histogram.h"
#include <set>
#include <unordered_set>

typedef std::vector<cv::Point> Polygon;
typedef std::vector<Polygon> Zone;

class CCTVsystem
{
	//matrices used to store information 
	cv::Mat background,
		intrusiveIm,
		canny,
		diffrgb,
		barrierImage,
		trainImage,
		detected;

	//crop rectangle used to remove black borders 
	cv::Rect crop;

	//detected objects
	std::vector<Polygon> objects;

	//all the zones of interest with and without train
	Zone 
		enteringZone, leavingZone, railwayZone,
		enteringZone_wt,railwayZone_wt,leavingZone_wt, 
		barrierZone;

	/**
	 * Find a threshold for foreground extraction based on the histogram h
	 * @param minDiff the minimum threshold for each channel
	 * @param maxi the maximum value of the histogram
	 * @param mini the minimum value of the histogram
	 */
	int extractThreshold(const Histogram& h, int minDiff, int maxi, int mini);

	/**
	 * Constraint run length algorithm (here used to fill contours from canny)
	 * @param input a binary image to fill
	 * @param filled the output image 
	 * @param Cx the maximum gap on the x-axis
	 * @param Cy the maximum gap on the y-axis
	 */
	void CRLA(const cv::Mat& input, cv::Mat& filled, int Cx, int Cy);

	/**
	 * Function used to select multiple polygons on an image and save them in zone
	 */
	int selectZone(Zone& zone, const char* name, const cv::Mat& img);

public:

	//booleans updated each time run(image) is called
	bool barrierDetected, enteringDetected, leavingDetected, ontrackDetected,trainDetected;

	/**
	 * Constructor, if the zones are not specified, a gui will open to select them
	 */
	CCTVsystem(const cv::Mat& bg, const cv::Mat& _trainImage, const cv::Mat& _barrierImage,
		const Zone& _enteringZone, const Zone& _leavingZone, const Zone& _railwayZone,
		const Zone& _enteringZone_wt, const Zone& _leavingZone_wt, const Zone& _railwayZone_wt, 
		const Zone& _barrierZone,
		const cv::Rect _crop = cv::Rect(cv::Point(23, 29), cv::Point(696, 463)));

	/**
	 * Extract a foreground mask from a RGB image using a histogram thresholding techniques
	 * @param img the input image (should be an image with mean close to zero)
	 * @param fg_mask the output binary foreground mask
	 * @param openingSize the size of the kernel used for opening 
	 * @param minDiff the minimum threshold on each channel of img
	 */
	void HistogramForegroundExtraction(const cv::Mat& img, cv::Mat& fg_mask, int openingSize, int min_diff);

	/**
	 * Extract a foreground mask using canny edge detection and the CRLA algorithm
	 * @param img the input image 
	 * @param fg_mask the output binary foreground mask
	 * @param low_thresh the low threshold for canny 
	 * @param high_thresh  the high threshold for canny
	 * @param blur_size the size of the gaussian kernel used to blur img before canny
	 * @param crlax,crlay the parameters for the CRLA algorithm
	 */
	void CannyForegroundExtraction(const cv::Mat& img, cv::Mat& fg_mask, int low_thresh, int high_thresh, int blur_size,int crlax, int crlay);

	/**
	 * Remove the shadows of the objects in the mask 
	 * @param mask , a foreground mask 
	 * @param background the background image
	 * @param image the actual image 
	 */
	void ShadowRemoval(cv::Mat& mask, const cv::Mat & background, const cv::Mat& image);

	/**
	 * Detect if the barrier is in the picture using probabilistic houghline
	 */
	bool DetectBarrier(const cv::Mat& img,float threshold);

	/**
	 * Main function that does the analysis of the scene using the other methods of the class
	 * @param image is the image to process
	 * @param minObjectSize is the minimum object size to consider
	 * @param trainMinArea is the threshold to change zones of interest
	 * @param background_update if true actualise the background when an empty image is found 
	 */
	void run(const cv::Mat& image, int minObjectSize, int trainMinArea, bool background_update);

	//other run functions which call the above one multiple times
	void run(const char* path);
	void run(cv::VideoCapture& cap);
	
	~CCTVsystem() {};
};

/**
 * Mouse callback to find x and y position of a mouse click
 */
void pointSelector(int event, int x, int y, int flags, void * userdata);
