#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include"ScreenVideoCapture.h"
#include<ctime>
#include<chrono>

#include<iostream>
#include<conio.h>           // may have to modify this line if not using Windows
#include<time.h>

#define SHOTS_NUMBER 100
#define PATH ""
#define WIDTH_SHRINKING_RATIO 0.5
#define HEIGHT_SHRINKING_RATIO 0.5


///////////////////////////////////////////////////////////////////////////////////////////////////

std::string ExePath() {//get execution path
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	ScreenVideoCapture svc = ScreenVideoCapture::ScreenVideoCapture(0);//value represents which screen to capture
	std::string archive_path = ExePath() + "\\/archive\\/";//saving image list to archive folder
	CreateDirectory(archive_path.c_str(), NULL);//create arcive folder if doesn't exist
	std::cout<<"writing captured images to: " + archive_path;


	cv::Mat img_raw_screen;
	cv::Mat img_Grayscale;
	cv::Mat img_Resized;

	std::chrono::milliseconds epoch_time_ms; //time since epoch in milliseconds

	svc.open(0);
	svc.read(img_raw_screen);

	if (img_raw_screen.empty()) {                                  // if unable to open image
		std::cout << "error: wasn'nt able to load image" << std::endl << std::endl ;    
		_getch();                                               // may have to modify this line if not using Windows
		return(-1);                                              // and exit program
	}

	//cv::imshow("raw_screen", img_raw_screen);
	
	cv::cvtColor(img_raw_screen, img_Grayscale, CV_BGR2GRAY);       // convert to grayscale
	//cv::imshow("img_Grayscale", img_Grayscale);
	cv::resize(img_Grayscale, img_Resized, cv::Size(), WIDTH_SHRINKING_RATIO, HEIGHT_SHRINKING_RATIO, cv::INTER_AREA);// usually best interpolation for shrinking
	//cv::resize(img_Grayscale, img_Resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);//fastest interpolation, (also the default one)
	//cv::imshow("img_Resized", img_Resized);

	for (int i = 0; i < SHOTS_NUMBER; i++) {
		svc.read(img_raw_screen);

		if (img_raw_screen.empty()) {                                  // if unable to open image
			std::cout << "error: wasn'nt able to load image" << std::endl << std::endl;
			_getch();                                               // may have to modify this line if not using Windows
			return(-1);                                              // and exit program
		}

		cv::cvtColor(img_raw_screen, img_Grayscale, CV_BGR2GRAY);       // convert to grayscale
		cv::resize(img_Grayscale, img_Resized, cv::Size(), WIDTH_SHRINKING_RATIO, HEIGHT_SHRINKING_RATIO, cv::INTER_AREA);//interpolation usually best for shrinking
		epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());//get current epoch time
		cv::imwrite(archive_path + std::to_string(epoch_time_ms.count()) +  ".jpg", img_Resized);
	}


	cv::waitKey(0);
	return(0);
}



