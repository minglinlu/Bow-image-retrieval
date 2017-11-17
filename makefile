CFLAGS = -std=c++11
#CFLAGS = -lstdc++

INCPATH = -I/usr/local/include
LIBPATH = -L/usr/local/lib

objects=train.o BagOfFeature.o
train:	$(objects)
	g++ $(CFLAGS) $(INCPATH) $(LIBPATH) -o train ${objects} `pkg-config --libs opencv` `pkg-config --cflags opencv`
train.o:train.cpp train.h
	g++ $(CFLAGS) $(INCPATH) -c train.cpp 
BagOfFeature.o:	BagOfFeature.cpp BagOfFeature.h
	g++ $(CFLAGS) $(INCPATH) -c BagOfFeature.cpp 

clean:
	rm -rf log *.log *.o *.out *.d train
