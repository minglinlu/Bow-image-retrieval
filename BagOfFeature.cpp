#include "BagOfFeature.h"

double BagOfFeature::cal_theta (Mat A, Mat B)
{
	double ab = A.dot (B);
	double aa = A.dot (A);
	double bb = B.dot (B);

	if (!aa) { return 0; }
	if (!bb) { return 0; }
	return ab / sqrt (aa * bb);
}

bool BagOfFeature::ReadImages(const string& imagesDir, vector<Mat>& images)
{
	try{
		for (int i = 0; i < 1000; i++){
			Mat img = imread(imagesDir + to_string(i)+".jpg");
			images.push_back(img);
		}
	}
	catch (exception &e){
		cout<<e.what()<<endl;
		return false;
	}
	return true;
}

Mat BagOfFeature::extractFeature (vector<Mat> &images)
{
	//input image and output features
	Mat input,features;
	//keypoints
	vector<KeyPoint> keypoints;
	Mat descriptor;
	//detector
	ORB orb;
	SiftFeatureDetector siftDetector;
	SurfFeatureDetector surfDetector(2000,4);
	//extractor
	FREAK freakExtractor;
	for (int f = 0; f < images.size(); f += 1){
		input = images[f];
		surfDetector.detect(input, keypoints);
		freakExtractor.compute(input, keypoints, descriptor);
		descriptor.convertTo(descriptor, CV_32FC1);
		features.push_back(descriptor);
	}
	return features;
}

Mat BagOfFeature::trainFeature (Mat &descriptors,int dictionarySize){
	TermCriteria tc (CV_TERMCRIT_ITER, 100, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	BOWKMeansTrainer bowTrainer (dictionarySize, tc, retries, flags);
	Mat dictionary = bowTrainer.cluster (descriptors);
	dictionary.convertTo(dictionary, CV_8UC1);
	FileStorage fs ("./data/surf_freak_dictionary_"+to_string(dictionarySize)+".yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	return dictionary;
}
