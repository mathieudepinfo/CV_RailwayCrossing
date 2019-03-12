/**
Computer Vision Assignement : Railway crossing monitoring
Author : MARSOT Mathieu

The folder already contained an existing build of the project
but you may want to rebuild it
To rebuild the solution run cmake with the x64 compiler platform setting.
Then build the ALL_BUILD solution to get the executable in build/bin/Debug(Release)

To launch the executable, specify a path as command line argument 
to run the program on multiple files. 

Example : ./ComputerVision.exe C:/path/to/image/test/files/*png

In the main file, possibility to uncomment the VideoCapture example
to test the program on a video

In the CCTVSystem.cpp file, possibility to uncomment some imshow functions 
to see how the object detection is proceeding

*/

#include "ComputerVision.h"

int main(int argc, char** argv)
{
	//reference images for background and barrier and train zone selection
	Mat a = imread("../../background.png",CV_LOAD_IMAGE_COLOR);
	Mat b = imread("../../barrier.png", CV_LOAD_IMAGE_COLOR);
	Mat c = imread("../../train.png", CV_LOAD_IMAGE_COLOR);

	if (a.empty() || b.empty() || c.empty()) {
		std::cout << "wrong path for reference images\n";
		return 1;
	}
	//Predefined zones, can probably be changed for better performance
	Zone entering({
		Polygon({Point(5,117),Point(89,193),Point(212,141),Point(38,33),Point(6,37)}),
		Polygon({Point(618,305),Point(669,344),Point(667,231)})
		});
	Zone leaving({
		Polygon({Point(189,44),Point(347,117),Point(382,95),Point(227,39)}),
		Polygon({Point(568,340),Point(437,426),Point(669,428),Point(666,391)})
		});
	Zone railway({
		Polygon({Point(161,342),Point(265,432),Point(666,167),Point(579,120)})
		});
	Zone entering_wt({
		Polygon({Point(37,206),Point(184,145),Point(57,36),Point(4,40),Point(3,167)}),
		Polygon({Point(581,329),Point(671,391),Point(669,268)})
		});
	Zone leaving_wt({
		Polygon({Point(179,44),Point(250,94),Point(304,75),Point(218,38)}),
		Polygon({Point(549,337),Point(413,425),Point(667,429),Point(667,381)})
		});
	Zone railway_wt({
		Polygon({Point(277,429),Point(197,428),Point(55,221),Point(515,33),Point(642,185)})
		});
	Zone barrier({
		Polygon({Point(2,224),Point(2,276),Point(342,148),Point(332,119)})
		});

	CCTVsystem cctv(a,c,b,entering,leaving, railway, entering_wt, leaving_wt, railway_wt, barrier);


	if(argc == 2) cctv.run(argv[1]);
	else std::cout << "Wrong arguments should be a path to multiple images \n";

	///Example of the system on a video
	/*cctv.run("C:/Users/mathi/Documents/CranfieldAssignement/ComputerVision/Individual/all/*.png");*/

	/*VideoCapture cap("../../levelcrossing.mpg"); 
	if (!cap.isOpened()) {
		cout << "cannot open video";
		return 1;
	}
	cctv.run(cap);*/
	
	return 0;
}
