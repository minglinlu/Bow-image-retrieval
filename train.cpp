#include "train.h"
#include "BagOfFeature.h"

int main(){
	auto_ptr<BagOfFeature> bow(new BagOfFeature());	
	vector<cv::Mat> images;
	//Read Images
	if(!bow->ReadImages("/Users/lml/Desktop/image.orig/",images)){
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
	Mat dictionary=bow->trainFeature(descriptors,100);
	cout<<"Training codebook end. Time used:"<<(double)(clock()-last_time)/CLOCKS_PER_SEC<<" sec."<<endl;
	return 0;
}
