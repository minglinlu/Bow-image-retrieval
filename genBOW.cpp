#include "BagOfFeature.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv){
	if (argc != 3) {
		cout<<"please input ./genBOW ./data/xxx.yml /xx/xx/"<<endl;
		return -1;
	}
	string dictDir=argv[1];
	string imagesDir=argv[2];
	auto_ptr<BagOfFeature> bow(new BagOfFeature());	
	//load the codebook here
	Mat dictionary=bow->loadDictionay(dictDir);
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
	bow->genBOWMap(dictionary,images);
	return 0;
}
