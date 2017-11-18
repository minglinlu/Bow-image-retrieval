#include "BagOfFeature.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv){
	if (argc != 2) {
		cout<<"please input ./genBOW /xx/xx/"<<endl;
		return -1;
	}
	//string dictDir=argv[1];
	//int num=stoi(argv[2]);
	string imagesDir=argv[1];
	cout<<imagesDir<<endl;
	auto_ptr<BagOfFeature> bow(new BagOfFeature());	
	//load the codebook here
	Mat dictionary=bow->loadDictionay("./data/dictionary.yml");
	cout<<"dictionary size: "<<dictionary.size()<<endl;

	//read images
	vector<cv::Mat> images;
	//if(!bow->readImages("/Users/lml/Desktop/image.orig/",images)){
	cout<<imagesDir<<endl;
	if(!bow->readImages(imagesDir,images)){
		cout<<"read images failed"<<endl;
	}
	cout<<"read "<<images.size()<<" images"<<endl;

	cout<<"genBOW now..."<<endl;
	clock_t last_time=clock();
	bow->genBOWMap(dictionary,images);
	cout<<" Time used:"<<(double)(clock()-last_time)/CLOCKS_PER_SEC<<"sec."<<endl;
	return 0;
}
