CFLAGS = -std=c++11

LIBS =  -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -lopencv_calib3d -lopencv_contrib

INCPATH = -I/usr/local/include
LIBPATH = -L/usr/local/lib

all:	trainDic
trainDic:	trainDic.cpp trainDic.h
	g++ $(CFLAGS) $(INCPATH) $(LIBPATH) -o train trainDic.cpp $(LIBS)

clean:
	rm -rf *.0 *.out *.d
