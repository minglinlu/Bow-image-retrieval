#ifndef _BagOfFeature_H

#define _BagOfFeature_H
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class BagOfFeature
{
	public:
		double cal_theta (Mat A, Mat B);
		bool ReadImages(const string& dir, vector<Mat>& images);
		Mat extractFeature (vector<Mat> &images);
		Mat genFeatureMap (const string &dictionaryDir, vector<Mat> &images);
		Mat genSingleFeatureMap(const string &dictionaryDir, vector<Mat> &images);
		Mat trainFeature (Mat &descriptors,int dictionarySize=100);
	private:
		clock_t last_time;
};

#endif
