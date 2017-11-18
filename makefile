CFLAGS = -std=c++11
#CFLAGS = -lstdc++

INCPATH = -I/usr/local/include
LIBPATH = -L/usr/local/lib

#objects=train.o BagOfFeature.o
all:	train genBOW query
query:query.o BagOfFeature.o
	g++ $(CFLAGS) $(INCPATH) $(LIBPATH) -o query query.o BagOfFeature.o `pkg-config --libs opencv` `pkg-config --cflags opencv`
query.o:query.cpp
	g++ -c query.cpp
genBOW:genBOW.o BagOfFeature.o
	g++ $(CFLAGS) $(INCPATH) $(LIBPATH) -o genBOW genBOW.o BagOfFeature.o `pkg-config --libs opencv` `pkg-config --cflags opencv`
genBOW.o:genBOW.cpp
	g++ -c genBOW.cpp
train:	train.o BagOfFeature.o
	g++ $(CFLAGS) $(INCPATH) $(LIBPATH) -o train train.o BagOfFeature.o `pkg-config --libs opencv` `pkg-config --cflags opencv`
train.o:train.cpp
	g++ $(CFLAGS) $(INCPATH) -c train.cpp 
BagOfFeature.o:	BagOfFeature.cpp BagOfFeature.h
	g++ $(CFLAGS) $(INCPATH) -c BagOfFeature.cpp 

clean:
	rm -rf log *.log *.o *.out *.d train genBOW query
