#include "BagOfFeature.h"

Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");	
Ptr<FeatureDetector> surfDetector(new SurfFeatureDetector(2000, 4));
Ptr<DescriptorExtractor> freakExtractor = DescriptorExtractor::create("FREAK");
BOWImgDescriptorExtractor bowDE (freakExtractor, matcher);

double BagOfFeature::cal_hist (Mat A, Mat B)
{
	CV_Assert(A.cols==B.cols&&A.rows==1&&B.rows==1);
	long cols=A.cols;
	Mat C=A.clone();
	for(int i=0;i<cols;i++){
		C.at<float>(0,i)=A.at<float>(0,i)<B.at<float>(0,i)?A.at<float>(0,i):B.at<float>(0,i);
	}
	Mat out(1,1,CV_32F,Scalar(0));
	reduce(C,out,1,CV_REDUCE_SUM);
	double ret=out.at<float>(0,0);
	return ret;
}
double BagOfFeature::cal_theta (Mat A, Mat B)
{
	double ab = A.dot (B);
	double aa = A.dot (A);
	double bb = B.dot (B);

	if (!aa) { return 0; }
	if (!bb) { return 0; }
	return ab / sqrt (aa * bb);
}

bool BagOfFeature::readImages(const string& imagesDir, vector<Mat>& images)
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

bool BagOfFeature::readImages(const string& imagesDir,int num, vector<Mat>& images)
{
	for (int i = 0; i < num; i++){
		Mat img = imread(imagesDir + to_string(i)+".jpg");
		images.push_back(img);
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

Mat BagOfFeature::genSingleBOWMap( Mat &dictionary,Mat &images){
	Mat bow;
	return bow;
}

Mat BagOfFeature::genBOWMap(Mat &dictionary, vector<Mat> &images){
	Mat bows;
	//	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");	
	//	Ptr<FeatureDetector> surfDetector(new SurfFeatureDetector(2000, 4));
	//	Ptr<DescriptorExtractor> freakExtractor = DescriptorExtractor::create("FREAK");
	//	BOWImgDescriptorExtractor bowDE (freakExtractor, matcher);
	bowDE.setVocabulary (dictionary);

	//where the bows to store
	FileStorage fs1 ("./data/surf_freak_bows_"+to_string(dictionary.rows)+".yml", FileStorage::WRITE);
	Mat input;
	Mat bowDescriptor;
	//idf=sum(every bow)
	Mat idf(1,dictionary.rows,CV_32F,Scalar(0));
	vector<KeyPoint> keypoints;
	for (int f = 0; f < images.size(); f += 1){
		input = images[f];
		surfDetector->detect (input, keypoints);
		bowDE.compute (input, keypoints, bowDescriptor);
		bowDescriptor *= keypoints.size();
		bowDescriptor.convertTo (bowDescriptor, CV_32F);
		//store original occurence of word i
		idf+=bowDescriptor/bowDescriptor;
		//Mat out(1,1,CV_32F,Scalar(0));
		//reduce(bowDescriptor,idf,0,CV_REDUCE_SUM);
		//cout<<out<<endl;
		bows.push_back (bowDescriptor);
	}
	//tf*idf;idf=log(D/n);each bow should be multipy 
	Mat idf1(1,dictionary.rows,CV_32F,Scalar(1));
	idf+=idf1;
	idf=(idf1/idf)*images.size();
	log(idf,idf);
	for(int i=0;i<bows.rows;i++){
		bows.row(i)=bows.row(i).mul(idf);
		normalize(bows.row(i),bows.row(i),1.0,0.0,NORM_L1);//L2 nomalize
		//Mat out(1,1,CV_32F,Scalar(0));
		//reduce(bows.row(i),out,1,CV_REDUCE_SUM);
		//cout<<out.at<float>(0,0)<<endl;
	}
	fs1 << "descriptor" << bows << "idf" << idf;
	fs1.release();
	return bows;
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
	fs.release();
	return dictionary;
}

Mat BagOfFeature::loadDictionay(const string &dictDir){
	Mat dictionary;
	FileStorage fs (dictDir, FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();
	return dictionary;
}

Mat BagOfFeature::loadBows(const string &bowsDir,Mat &idf){
	Mat bows;
	FileStorage fs (bowsDir, FileStorage::READ);
	fs["descriptor"] >> bows;
	fs["idf"] >> idf;
	//cout<<idf<<endl;
	fs.release();
	return bows;
}

//  int CmpByValue(const pair<int, double>& lhs, const pair<int, double>& rhs) {  
//    return lhs.second > rhs.second;  
//  }  
vector<Mat> BagOfFeature::queryImg(Mat &queryImg,Mat &dictionary,Mat &bows,Mat &idf){
	vector<Mat> retImgs;
	bowDE.setVocabulary (dictionary);

	//cal the bow for query image
	vector<KeyPoint> keypoints;
	Mat bowDescriptor(1,dictionary.rows,CV_32F,Scalar(0));
	surfDetector->detect (queryImg, keypoints);
	bowDE.compute (queryImg, keypoints, bowDescriptor);
	cout<<keypoints.size()<<endl;
	if(keypoints.size()>0)bowDescriptor *= keypoints.size();
	else{bowDescriptor=Mat(1,dictionary.rows,CV_32F,Scalar(0));}
	bowDescriptor.convertTo (bowDescriptor, CV_32F);
	bowDescriptor=bowDescriptor.mul(idf);
	normalize(bowDescriptor,bowDescriptor,1,0,NORM_L1);

	//cout<<bowDescriptor<<endl;
	//cout<<idf<<endl;
	//compare
	double max=-1;int id=-1;
	for(int i=0;i<bows.rows;i++){
		double score=cal_hist(bows.row(i),bowDescriptor);
		//cout<<score<<endl;
		if(score>max){
			max=score;
			id=i;
			cout<<id<<".jpg sim: "<<max<<endl;
		}
	}
	//cout<<id<<".jpg sim: "<<max<<endl;
	//sort(rank.begin(),rank.end(),CmpByValue());
	return retImgs;
}
