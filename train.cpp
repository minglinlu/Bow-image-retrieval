#include "BagOfFeature.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv){
	int dictionarySize=100;
	if (argc == 2) {
		dictionarySize=stoi(argv[1]);
		cout<<"dicitionary size is : "<<dictionarySize<<endl;
	}
	auto_ptr<BagOfFeature> bow(new BagOfFeature());	
	vector<cv::Mat> images;
	//Read Images
	if(!bow->readImages("/Users/lml/Desktop/image.orig/",images)){
		cout<<"read images failed"<<endl;
	}
	cout<<"read "<<images.size()<<" images"<<endl;

	//Extract Features
	auto last_time=clock();
	cout<<"extract features begin..."<<endl;
	Mat descriptors=bow->extractFeature(images);
	cout<<"extract features end. Time used:"<<(double)(clock()-last_time)/CLOCKS_PER_SEC<<" sec."<<endl; 
	cout<<"Load "<<descriptors.size()<<" feature points"<<endl;

	cout<<"Start training descriptors..."<<endl;
	last_time=clock();
	Mat dictionary=bow->trainFeature(descriptors,dictionarySize);
	cout<<"Training codebook end. Time used:"<<(double)(clock()-last_time)/CLOCKS_PER_SEC<<" sec."<<endl;
	return 0;
}
