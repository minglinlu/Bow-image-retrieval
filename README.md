# Bow-image-retrieval
Bow-image-retrieval

# Created by luminglin on 2017.11.17
1.extract key points and descriptors

2.generate BOWs for each image using SURF detector and FREAK descriptors.

3.complete idf computation.

# steps to use
You need to install opencv and make sure the makefile is no problem.
And get the images set from http://wang.ist.psu.edu/~jwang/test1.tar, tar -zxvf to you local dir.
vi train.cpp and replace "/Users/lml/Desktop/image.orig/" to your local dir.

Then try 'make'...
We can get 3 output files: train genBOW query

There are 3 steps to do now.

(1) ./train 1000
train the dictionary with k=1000

(2) ./genBOW /Users/lml/Desktop/image.orig/
generate the BOW Mat for every image, incluing the idf.

(3) ./query /Users/lml/Desktop/image.orig/999.jpg
Now, you can get the result in ./result.html, open result.html, you can get the rank result.

# Thanks 
