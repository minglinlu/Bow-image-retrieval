#include "BagOfFeature.h" 
#include <iostream>

using namespace std;

int main(int argc, char **argv){
	if(argc !=2){
		cout<<"please input ./query /x/x/x/x.jpg"<<endl;
		return -1;
	}
	string queryImg=argv[1];
	auto_ptr<BagOfFeature> bow(new BagOfFeature());
	
	//load the codebook here
	string dictDir="/Users/lml/Desktop/CPP/Bow-image-retrieval/data/surf_freak_dictionary_1000.yml";
	Mat dictionary=bow->loadDictionay(dictDir);
	cout<<"dictionary size: "<<dictionary.size()<<endl;

	Mat idf;
	Mat bows=bow->loadBows("/Users/lml/Desktop/CPP/Bow-image-retrieval/data/surf_freak_bows_1000.yml",idf);
	cout<<"bows size:"<<bows.size()<<endl;
	Mat img = imread(queryImg);

	clock_t last_time=clock();
	vector<Mat> result= bow->queryImg(img,dictionary,bows,idf);
	cout<<"retrieval time used :"<<(double)(clock()-last_time)/CLOCKS_PER_SEC<<endl;
	return 0;
}
