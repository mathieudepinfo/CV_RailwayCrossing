#include "Histogram.h"



Histogram::Histogram(const cv::Mat& m, int maxi, int mini)
	:cv::Mat(3, maxi - mini + ((maxi > 0 && mini <= 0) ? 1 : 0), CV_32FC1), maxV(maxi), minV(mini)
{

	//nob = number of bars if 0 included 1 more value
	nob = maxi - mini + ((maxi > 0 && mini <= 0) ? 1 : 0);

	if (m.channels() != 3) throw "RGB only";

	(*this).setTo(0.0f);
	for (int i = 0; i < m.rows*m.cols; ++i) {
		for (int c = 0; c < 3; ++c) {
			(*this).at<float>(c, cvRound(m.at<cv::Vec3s>(i / m.cols, i%m.cols)[c])-mini) += 1.0f;
		}
	}


};


Histogram::~Histogram()
{
}

void Histogram::normalize(float min, float max)
{
	cv::normalize((*this), (*this), min, max, cv::NORM_MINMAX, CV_32FC1);
}

void Histogram::draw(const char* winName) {

	int hist_w = 1024, hist_h = 400;
	int bin_w = cvRound((double)hist_w / nob);
	Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	std::vector<cv::Mat> channels(3);

	for (int c = 0;c< 3;++c){

		channels[c] = (*this)(cv::Rect(0,c, this->cols,1)).clone();
		cv::normalize(channels[c], channels[c], 0, histImage.rows, cv::NORM_MINMAX, -1, Mat());
	}

	cv::Scalar clrs[3] = { cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 0),cv::Scalar(0, 0, 255) };

	/// Draw for each channel
	for (int i = 1; i < nob; i++)
	{
		for (size_t c = 0; c < 3; ++c) {
			
			line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(channels[c].at<float>(0,i - 1))),
				cv::Point(bin_w*(i), hist_h - cvRound(channels[c].at<float>(0,i))),
				clrs[c], 1, 8, 0);
		}
	
	}

	cv::imshow(winName,histImage);
}

float Histogram::Histogram::get(int channel, int value) const
{
	return (*this).at<float>(channel,value-minV);
}

void Histogram::smooth(int k)
{
	Mat tmp = (*this).clone();

	for (int b = k; b < nob - k; ++b) {

		for (int c = 0; c < 3; ++c) {
			for (int i = -k; i <= k; ++i) {
				(*this).at<float>(c, b) += tmp.at<float>(c,b+i);
			}
			(*this).at<float>(c, b) /= 2.0f*k+1.0f;
		}
	}
	
}

void Histogram::set(int channel, int value,float newVal)
{
	(*this).at<float>(channel, value - minV) = newVal;
}

cv::Mat Histogram::Histogram::getChannel(int channel) const
{
	return (*this)(cv::Rect(0,channel,nob,1));
}