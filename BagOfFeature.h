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
		double cal_hist (Mat A, Mat B);
		bool readImages(const string& dir, vector<Mat>& images);
		bool readImages(const string& dir,int num, vector<Mat>& images);
		Mat extractFeature (vector<Mat> &images);
		Mat genBOWMap (Mat &dictionary, vector<Mat> &images);
		Mat genSingleBOWMap( Mat &dictionary,Mat &images);
		Mat trainFeature (Mat &descriptors,int dictionarySize=100);
		Mat loadDictionay(const string &dictDir);
		Mat loadBows(const string &bowsDir,Mat &idf);
		vector<pair<string,float> > queryImg(Mat &queryImg,Mat &dictionary,Mat &bows,Mat &idf);
	private:
		clock_t last_time;
};

#endif
