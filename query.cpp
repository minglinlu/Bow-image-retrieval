#include "BagOfFeature.h" 
#include <iostream>
#include <fstream>

using namespace std;

void writeHtml(vector<pair<string,float> > result,string queryImg){
	ofstream ofs("result.html");
	if (!ofs) throw runtime_error("Cannot open file.");
	ofs << "<html><head><style> img {width:250px; border:0px; margin:5px 5px; padding:0px 0px;} .divcss5{text-align:center} </style></head><body><div class=\"divcss5\"><h2>Query</h2><div><img src=\"" << queryImg << "\" alt=\"\" /><br />";
	ofs << "<div class=\"divcss5\"><h2>Result</h2></div><table><tbody>";
	ofs <<"<div align='center' style='margin-left:auto;margin-right:auto'>";
	for (vector<pair<string,float> >::reverse_iterator iter=result.rbegin();iter!=result.rend();iter++) {
		ofs << "<div style='float:left'>" <<iter->second<<"<br>";
		ofs << "<img alt=\""<<iter->second<<"\" src=\"" << iter->first << "\" />";
		ofs <<"</div>";
	}
	ofs <<"</div>";
	ofs << "</tbody></table></body></html>";
	ofs.close();
	cerr << "Html outputted." << endl;
}

int main(int argc, char **argv){
	if(argc !=2){
		cout<<"please input ./query /x/x/x/x.jpg"<<endl;
		return -1;
	}
	string queryImg=argv[1];
	auto_ptr<BagOfFeature> bow(new BagOfFeature());

	//load the codebook here
	string dictDir="/Users/lml/Desktop/CPP/Bow-image-retrieval/data/dictionary.yml";
	Mat dictionary=bow->loadDictionay(dictDir);
	cout<<"dictionary size: "<<dictionary.size()<<endl;

	Mat idf;
	Mat bows=bow->loadBows("/Users/lml/Desktop/CPP/Bow-image-retrieval/data/bows.yml",idf);
	cout<<"bows size:"<<bows.size()<<endl;
	Mat img = imread(queryImg);

	clock_t last_time=clock();
	vector<pair<string,float> > result= bow->queryImg(img,dictionary,bows,idf);
	writeHtml(result,queryImg);
	cout<<"retrieval time used :"<<(double)(clock()-last_time)/CLOCKS_PER_SEC<<endl;
	return 0;
}
