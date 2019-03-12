#include "CCTVsystem.h"

int CCTVsystem::extractThreshold(const Histogram & hi, int minDiff, int maxi, int mini)
{
	//Initialier thresholds to a minimum difference, can be 0
	int TH_UP[3] = { minDiff,minDiff,minDiff };
	int TH_LOW[3] = { -minDiff,-minDiff,-minDiff };

	Histogram h(hi);

	//find the highest peak close to zero for each channel
	int highest[3] = { 0,0,0 };
	for (int c = 0; c < 3; ++c) {
		for (int i = -30; i < 30 - 1; ++i) {
			if (h.get(c, highest[c]) < h.get(c, i)) {
				highest[c] = i;
			}
		}

	}
	
	//smoothing histogram to reduce noise problems
	h.smooth(2);
	
	//for each color, finding the turning points
	for (int c = 0; c < 3; ++c) {
		for (int i = highest[c]+minDiff; i < maxi - 1; ++i) {
			if (h.get(c, i - 1) >= h.get(c, i) && h.get(c, i) <= h.get(c, i + 1)) {
				TH_UP[c] = i;
				break;
			}
		}

		for (int i = highest[c]-minDiff; i >= mini; --i) {
			if (h.get(c, i + 1) >= h.get(c, i) && h.get(c, i) <= h.get(c, i - 1)) {
				TH_LOW[c] = i;
				break;
			}
		}
	}

	int SH = abs(TH_UP[H_RED]) + abs(TH_UP[H_GREEN]) + abs(TH_UP[H_BLUE]);
	int SL = abs(TH_LOW[H_RED]) + abs(TH_LOW[H_GREEN]) + abs(TH_LOW[H_BLUE]);

	//in the paper they use min(SH,SL) but true negatives are more welcome than false positives
	return std::max(SL, SH);
}

void CCTVsystem::CRLA(const cv::Mat & input,cv::Mat& filled, int Cx, int Cy)
{
	cv::Mat outputX = input.clone();
	cv::Mat outputY = input.clone();

	int start = 0, end = 0;

	//row loop to connect points horizontally
	for (int i = 0; i < input.rows; ++i) {
		start = 0;
		end = 0;

		//finding the first non zero element of a row
		while (start < input.cols && input.at<uchar>(i, start) == 0) start++;
		end = start;

		//while the last element of the row has not been reached
		while (start < input.cols) {
			//look for the next 255 in the row
			while (end < input.cols && input.at<uchar>(i, end) == 0) end++;

			//if it is not too far from the start, fill the gap with 255
			if ((end - start) < Cx && (end != input.cols - 1)) {
				for (int p = start; p < end; ++p) {
					outputX.at<uchar>(i, p) = 255;
				}
			}
			//restart the process for the next element
			start = end;
			end++;	
		}
	}

	//col loop to connect points vertically, same procedure as above
	for (int j = 0; j < input.cols; ++j) {

		start = 0;
		end = 0;

		while (start < input.rows && input.at<uchar>(start, j) == 0) start++;

		end = start;

		while (start < input.rows) {
			while (end < input.rows && input.at<uchar>(end, j) == 0) {
				end++;
			}

			if ((end - start) < Cy && (end != input.rows - 1)) {

				for (int p = start; p < end; ++p) {

					outputY.at<uchar>(p, j) = 255;
				}
			}

			start = end;
			end++;
		}
	}

	//and the two matrices to get output 
	cv::bitwise_and(outputY, outputX, filled);

	outputX.release();
	outputY.release();
}

void CCTVsystem::ShadowRemoval(cv::Mat & mask, const cv::Mat & background,const cv::Mat& image)
{
	cv::Mat background_grey,image_grey;

	if (background.channels() == 1) background_grey = background.clone();
	else cv::cvtColor(background, background_grey,cv::COLOR_BGR2GRAY);
	
	if (image.channels() == 1) image_grey = image.clone();
	else cv::cvtColor(image, image_grey, cv::COLOR_BGR2GRAY);

	cv::bitwise_and(image_grey, mask, image_grey);

	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			if (intrusiveIm.at<uchar>(i, j) != 0) {
				float q = static_cast<float>(background_grey.at<uchar>(i, j)) / static_cast<float>(image_grey.at<uchar>(i, j));
				if (1.0 < q && q < 1.6) {
					mask.at<uchar>(i, j) = 0;
				}
			}
		}
	}

	background_grey.release();
	image_grey.release();
}

bool CCTVsystem::DetectBarrier(const cv::Mat & img, float threshold)
{

	std::vector<cv::Vec4i> lines;
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0);
	cv::Canny(gray, gray, 100, 150);
	HoughLinesP(gray, lines, 1, CV_PI / 180, 50, 100, 20);
	cv::Mat linesI = cv::Mat::zeros(gray.size(),CV_8UC1);
	
	cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
			
		//if the line starts in the zone then it s probably the barrier
		// Further improvement : check if the detected line is black => shadow of the barrier
		if (cv::pointPolygonTest(barrierZone[0], cv::Point(l[0], l[1]), false) > 0
			&& cv::pointPolygonTest(barrierZone[0], cv::Point(l[2], l[3]), false) > 0) {
			cv::line(linesI, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 3, CV_AA);

			//checking if it is a shadow
			cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
			cv::line(mask, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 255, 255), 3, CV_AA);

			if (cv::mean(img, mask)[2] > threshold) return true;
					
		}
	}
	
	return false;
}


CCTVsystem::CCTVsystem(const cv::Mat & bg, const cv::Mat& _trainImage, const cv::Mat& _barrierImage,
	const Zone& _enteringZone, const Zone& _leavingZone, const Zone& _railwayZone,
	const Zone& _enteringZone_wt, const Zone& _leavingZone_wt, const Zone& _railwayZone_wt, 
	const Zone& _barrierZone,
	const cv::Rect _crop):background(bg(_crop).clone()), trainImage(_trainImage(_crop).clone()), barrierImage(_barrierImage(_crop).clone()),crop(_crop)
{
	//setting zones for detection
	enteringZone = _enteringZone;
	leavingZone = _leavingZone;
	railwayZone = _railwayZone;
	//zones wt = with train
	enteringZone_wt = _enteringZone_wt;
	leavingZone_wt = _leavingZone_wt;
	railwayZone_wt = _railwayZone_wt;

	barrierZone = _barrierZone;

	//allocating memory for intrusiveIm and canny
	intrusiveIm = cv::Mat::zeros(background.size(), background.type());
	canny = cv::Mat::zeros(background.size(), CV_8UC1);


	int x = 0;

	//if some zones are not specified select them with the mouse 
	std::string consigne(" (space for adding another polygon to the zone, enter to save the zone)");
	if (enteringZone.size() == 0) {
		//wait for enter key
		while (x != 13) {
			enteringZone.push_back(Polygon());
			std::string title("entering number " + std::to_string(enteringZone.size()));
			x = selectZone(enteringZone, (title + consigne).c_str(),background);
		}
	}

	x = 0;
	if (leavingZone.size() == 0) {
		while (x != 13) {
			leavingZone.push_back(Polygon());
			std::string title("leaving number " + std::to_string(leavingZone.size()));
			x = selectZone(leavingZone, (title + consigne).c_str(),background);
		}
	}
	x = 0;
	if (railwayZone.size() == 0) {
		while (x != 13) {
			railwayZone.push_back(Polygon());
			std::string title("railway number " + std::to_string(railwayZone.size()));
			x = selectZone(railwayZone, (title + consigne).c_str() , background);
		}
	}

	if (enteringZone_wt.size() == 0) {
		//wait for enter key
		while (x != 13) {
			enteringZone_wt.push_back(Polygon());
			std::string title("entering with train number " + std::to_string(enteringZone_wt.size()));
			x = selectZone(enteringZone_wt, (title + consigne).c_str(),trainImage);
		}
	}

	x = 0;
	if (leavingZone_wt.size() == 0) {
		while (x != 13) {
			leavingZone_wt.push_back(Polygon());
			std::string title("leaving with train number " + std::to_string(leavingZone_wt.size()));
			x = selectZone(leavingZone_wt, (title + consigne).c_str(),trainImage);
		}
	}
	x = 0;
	if (railwayZone_wt.size() == 0) {
		while (x != 13) {
			railwayZone_wt.push_back(Polygon());
			std::string title("railway with train number " + std::to_string(railwayZone_wt.size()));
			x = selectZone(railwayZone_wt, (title + consigne).c_str(),trainImage);
		}
	}

	x = 0;
	if (barrierZone.size()==0) {
		while (x != 13) {
			barrierZone.push_back(Polygon());
			std::string title("barrier number " + std::to_string(barrierZone.size()));
			x = selectZone(barrierZone, (title + consigne).c_str(),barrierImage);
		}
	}
	
}

void CCTVsystem::HistogramForegroundExtraction(const cv::Mat & img, cv::Mat & fg_mask, int openingSize, int min_diff)
{
	if (img.channels() != 3) throw "HistogramForegroundExtraction requires 3 channels input image";

	//normalize image to be more robust to different kind of input
	cv::Mat normalized_image = img.clone();
	double mini, maxi;
	cv::minMaxLoc(img, &mini, &maxi);
	if (mini<-255.0 || maxi >255.0) cv::normalize(img, normalized_image, -255, 255, cv::NORM_MINMAX);
	normalized_image.convertTo(normalized_image, CV_16SC3);

	//create histogram of normalized image
	Histogram h = Histogram(normalized_image);

	//extract threshold by looking for turning points of histogram
	int S = extractThreshold(h, min_diff, 255, -255);
	
	//if fg_mask is initialized reset to zero otherwise create new matrix
	if (fg_mask.size() != img.size()) fg_mask = cv::Mat::zeros(img.size(), CV_8UC1);
	else fg_mask.setTo(cv::Scalar(0));

	for (int i = 0; i < normalized_image.rows; ++i) {
		for (int j = 0; j < normalized_image.cols; ++j) {
			
			cv::Vec3s pixel = normalized_image.at<cv::Vec3s>(i, j);
			int norm = static_cast<int>(cv::norm(pixel,cv::NORM_L1));
			
			//if norm greater than threshold => intrusive object
			if (norm > S) {
				fg_mask.at<uchar>(i, j) = 255;
			}
		}
	}

	//opening to remove small isolated white pixels
	cv::Mat kernelOpening = cv::Mat::ones(openingSize, openingSize, CV_8UC1);
	cv::erode(fg_mask, fg_mask, kernelOpening);
	cv::dilate(fg_mask, fg_mask, kernelOpening);

	kernelOpening.release();
	normalized_image.release();
}

void CCTVsystem::CannyForegroundExtraction(const cv::Mat & img, cv::Mat & fg_mask, int low_thresh, int high_thresh,int blur_size, int crlax, int crlay)
{	
	//preparing blurred image for canny algorithm
	cv::Mat blurred_img;
	blurred_img = cv::abs(img);
	blurred_img.convertTo(blurred_img, CV_8UC3);
	cv::GaussianBlur(blurred_img, blurred_img, cv::Size(blur_size,blur_size), 0, 0);
	cv::Mat cannyPrep;
	
	cv::cvtColor(blurred_img, cannyPrep, cv::COLOR_BGR2GRAY);

	//canny algorithm, result save in canny attribute
	//cv::GaussianBlur(cannyPrep, cannyPrep, cv::Size(blur_size, blur_size), 0, 0);
	cv::Canny(cannyPrep, canny, low_thresh, high_thresh);

	//dilate contours before filling for better efficiency
	cv::Mat dilation_kernel = cv::Mat::ones(3, 3, CV_8UC1);
	dilation_kernel.at<uchar>(0, 0) = 0;
	dilation_kernel.at<uchar>(0, 2) = 0;
	dilation_kernel.at<uchar>(2, 0) = 0;
	dilation_kernel.at<uchar>(2, 2) = 0;

	cv::dilate(canny, canny, dilation_kernel);
	CRLA(canny, fg_mask, 100, 100);

	cannyPrep.release();
}

int CCTVsystem::selectZone(Zone& zone,const char* name,const cv::Mat& img) {

	cv::Mat bgc = img.clone();
	cv::imshow(name, bgc);

	cv::setMouseCallback(name, pointSelector
		, &(zone.back()));

	int x = 0;
	while (x != 32 && x!=13) {
		x = cv::waitKey(10);

		bgc.release();
		bgc = img.clone();
		//drawing all the subzones
		for (const Polygon& subzone : zone) {
			int i = 0;
			if (subzone.size() > 0) {
				
				while (i < subzone.size() - 1) {
					cv::line(bgc, subzone[i], subzone[i + 1], cv::Scalar(0, 0, 255));
					++i;
				}
				cv::line(bgc, subzone.back(), *subzone.begin(), cv::Scalar(0, 0, 255));

			}
			
		}
		cv::imshow(name, bgc);
	};

	bgc.release();
	cv::destroyWindow(name);
	return x;
}

void CCTVsystem::run(const cv::Mat& image,int minObjectSize,int trainMinArea,bool background_update) {
	if (image.empty())return;

	//foreground masks H = histogram, C = canny, fg_mask = fused mask
	cv::Mat fg_maskH(background.size(), CV_8UC1), fg_maskC(background.size(), CV_8UC1), fg_mask(background.size(), CV_8UC1);

	cv::Mat kernelClosing = cv::Mat::ones(11, 11, CV_8UC1);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point> > contours_poly;

	//removing the black borders
	cv::Mat im = image(crop).clone();

	//getting diference image
	diffrgb = cv::Mat(background.size(), CV_16SC3);
	cv::subtract(im, background, diffrgb, cv::Mat(), diffrgb.type());

	//creating foreground mask from difference image
	HistogramForegroundExtraction(diffrgb, fg_maskH, 3, 5);
	CannyForegroundExtraction(diffrgb, fg_maskC, 40, 70, 11, 100, 100);

	//remove shadows from detected objects
	ShadowRemoval(fg_maskH, background, im);

	//imshow("foreground mask histogram", fg_maskH);

	//and the two masks for more robust foreground extraction
	cv::bitwise_and(fg_maskC, fg_maskH, fg_mask);

	//imshow("foreground mask canny", fg_maskC);

	//close the obtain mask
	cv::dilate(fg_mask, fg_mask, kernelClosing);
	cv::erode(fg_mask, fg_mask, kernelClosing);

	//find contours of objects in the foreground mask
	contours.clear();
	cv::findContours(fg_mask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	cv::Mat drawing = cv::Mat::zeros(fg_mask.size(), CV_8UC3);

	contours_poly.clear();
	contours_poly.resize(contours.size());

	std::vector<cv::Point2f> center(contours.size());

	//associate each contour to a polygon for easy area calculation
	for (int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
	}

	cv::Mat dst = cv::Mat::zeros(im.size(), CV_8UC1);

	objects.clear();

	bool possibletrain = false;
	for (int i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(0, 0, 255);

		double area = cv::contourArea(contours_poly[i]);

		//small objects are not considered
		if (area > minObjectSize) {
			objects.push_back(contours_poly[i]);
		}
		if (area > trainMinArea) {
			possibletrain = true;
		}
	}

	cv::Mat final = cv::Mat::zeros(im.size(), CV_8UC1);

	//filling selected objects to obtain final foreground mask
	cv::fillPoly(final, objects, cv::Scalar(255));

	//close final mask for better labelling
	cv::dilate(final, final, kernelClosing);
	cv::erode(final, final, kernelClosing);

	//label objects so they are detected one time only
	cv::Mat labeled;
	int nl = cv::connectedComponents(final, labeled, 8);

	std::vector<std::vector<int>> cpt(nl, std::vector<int>(3));

	cv::Mat rdetect = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat enteringdetect = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat leavingdetect = cv::Mat::zeros(im.size(), CV_8UC1);

	//chose which zone to use (if there is a possible train the zone are changed)
	Zone& eZ = (possibletrain) ? enteringZone_wt : enteringZone;
	Zone& lZ = (possibletrain) ? leavingZone_wt : leavingZone;
	Zone& rZ = (possibletrain) ? railwayZone_wt : railwayZone;

	cv::fillPoly(enteringdetect, eZ, cv::Scalar(255));
	cv::fillPoly(leavingdetect, lZ, cv::Scalar(255));
	cv::fillPoly(rdetect, rZ, cv::Scalar(255));

	//use the zones as masks on the final image
	cv::bitwise_and(rdetect, final, rdetect);
	cv::bitwise_and(enteringdetect, final, enteringdetect);
	cv::bitwise_and(leavingdetect, final, leavingdetect);

	//for each zone count how many pixel of each label are detected
	for (int i = 0; i < final.rows; ++i) {
		for (int j = 0; j < final.cols; ++j) {
			int label = labeled.at<int>(i, j);
			if (rdetect.at<uchar>(i, j) == 255) cpt[label][0]++;
			if (enteringdetect.at<uchar>(i, j) == 255) cpt[label][1]++;
			if (leavingdetect.at<uchar>(i, j) == 255) cpt[label][2]++;
		}
	}

	
	trainDetected = false;

	for (int i = 0; i < final.rows; ++i) {
		for (int j = 0; j < final.cols; ++j) {

			//getting label of each pixel, if label=0 background so not interesting
			int label = labeled.at<int>(i, j);
			if (label == 0)continue;

			//confirming that it was a train
			if (cpt[label][0] > trainMinArea) trainDetected = true;

			//finding which zone has the more pixel of the selected label
			auto it = std::max_element(cpt[label].begin(), cpt[label].end());
			if (*it == 0) continue;

			//if it is the first element => on rail
			if (it == cpt[label].begin()) {
				rdetect.at<uchar>(i, j) = 255;
				if (enteringdetect.at<uchar>(i, j) == 255) enteringdetect.at<uchar>(i, j) = 0;
				if (leavingdetect.at<uchar>(i, j) == 255) leavingdetect.at<uchar>(i, j) = 0;
			}

			//if it is the second element => leaving
			else if (it == cpt[label].begin() + 1) {
				enteringdetect.at<uchar>(i, j) = 255;
				if (rdetect.at<uchar>(i, j) == 255) rdetect.at<uchar>(i, j) = 0;
				if (leavingdetect.at<uchar>(i, j) == 255) leavingdetect.at<uchar>(i, j) = 0;
			}
			else {//it if is the last one => entering
				leavingdetect.at<uchar>(i, j) = 255;
				if (rdetect.at<uchar>(i, j) == 255) rdetect.at<uchar>(i, j) = 0;
				if (enteringdetect.at<uchar>(i, j) == 255) enteringdetect.at<uchar>(i, j) = 0;
			}

		}
	}

	
	ontrackDetected = false;
	barrierDetected = false;
	enteringDetected = false;
	leavingDetected = false;

	//if an object is detected in a zone print it in console
	if (cv::norm(rdetect, cv::NORM_INF) > 0) {
		if (!trainDetected) ontrackDetected = true;
	}
	if (cv::norm(enteringdetect, cv::NORM_INF) > 0) {
		enteringDetected = true;
	}
	if (cv::norm(leavingdetect, cv::NORM_INF) > 0) {
		leavingDetected = true;
	}

	//the barrier detection doesn't work well if there is a train
	if (!trainDetected && DetectBarrier(im, 150.0f)) {
		barrierDetected = true;
	};

	//if everything false empty scene
	if (!(ontrackDetected || enteringDetected || leavingDetected || barrierDetected)) {
		if (background_update) {
			background = im;
		}
	}
	std::cout << std::endl;

	//nice display of the processed image
	std::vector<cv::Mat> v({ enteringdetect,leavingdetect,rdetect });
	
	cv::merge(v, detected);
	cv::addWeighted(detected, 0.5, im, 0.5, 0, detected);
	
	
}
void CCTVsystem::run(cv::VideoCapture& cap)
{
	cv::Mat frame;

	cap >> frame;

	int i = 0;
	while (!frame.empty()) {

		

		run(frame, 400, 60000, false);

		std::cout << i++ << ":";
		if (ontrackDetected) std::cout << "event 1 ";
		if (enteringDetected) std::cout << "event 2 ";
		if (leavingDetected) std::cout << "event 3 ";
		if (barrierDetected) std::cout << "event 4 ";
		if (trainDetected) std::cout << "event 5 ";
		//if nothing happened
		if (!(ontrackDetected || enteringDetected || leavingDetected || barrierDetected || trainDetected)) std::cout << "event 0 ";
		std::cout << "\n";
		//imshow("detected", detected);
		cap >> frame;
		//cv::waitKey(10);
	}
}

void CCTVsystem::run(const char * path)
{
	//list of images in the folder
	std::vector<cv::String> fn;
	cv::glob(path, fn, true);
 
	//matrices to store images and gray version of background used for shadow removal
	cv::Mat im;

	//for each image in the folder
	for (size_t k = 0; k < fn.size(); ++k)
	{
		//read image
		im = cv::imread(fn[k]);
		
		run(im, 400, 60000, false);
		std::cout << fn[k] << ":";
		if (ontrackDetected) std::cout << "event 1 ";
		if (enteringDetected) std::cout << "event 2 ";
		if (leavingDetected) std::cout << "event 3 ";
		if (barrierDetected) std::cout << "event 4 ";
		if (trainDetected) std::cout << "event 5 ";
		//if nothing happened
		if(!(ontrackDetected || enteringDetected || leavingDetected || barrierDetected || trainDetected)) std::cout << "event 0 ";
		std::cout << "\n";
		//cv::imshow("detected", detected);
		
		//cv::waitKey(10);
	}
}

void pointSelector(int event, int x, int y, int flags, void* userdata) {

	Polygon* p = reinterpret_cast<Polygon*>(userdata);

	if (event == cv::EVENT_LBUTTONUP) {
		p->push_back(cv::Point(x, y));
	}
}