#pragma once

#include <opencv2\opencv.hpp>
#include <Windows.h>

class ScreenVideoCapture
{
public:
	ScreenVideoCapture(int displayIndex = -1);
	~ScreenVideoCapture();

	void open(int displayIndex);
	void read(cv::Mat& destination);
	ScreenVideoCapture& operator>>(cv::Mat& destination);

private:
	cv::Rect2d captureArea;
	HWND targetWindow = NULL;


	void captureHwnd(HWND window, cv::Rect2d targetArea, cv::Mat& dest);
	static BOOL CALLBACK monitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData);
};

struct MonitorIndexLookupInfo
{
	int targetIndex;

	RECT outRect;
	int currentIndex;
};