# imageanalysis
Object Detection using OpenCV 3.2.0

This is a program written in C++ using OpenCV 3.2.0 to create an Support Vector Machine which is able to distinguish objects from each other. Our aim is to write code which is able to distinguish images containing garbage from images which dont contain garbage. We have taken the approach as described by skm on http://stackoverflow.com/questions/31414782/image-classification-in-opencv-python-based-on-training-set. However this approach needed to be tweaked quite a bit to mould it into the new OpenCV 3.2.0 library. This code was executed successfully on a Windows 10 machine running Visual Studio 2015 with all the latest updates installed. The library was compiled seperately from the source with the inclusion of the additional opencv-contrib package to make use of the non-free algorithms. A clear description of the approach has been given in the main source code in the form of comments. 

We have added a tutorial document for newbies to refer to for installation guidelines. Check out Instructions for installation of OpenCV.txt for more. It contains instructions to compile OpenCV from source and get started with your own Visual Studio C++ project.

Instructions for contribution:
The following instructions for contribution to this repository are only for those users who want to use Visual Studio for the development of this application. Note that when you wish to create a branch DO NOT branch the master branch. Instead branch the config branch and open up the properties of your project in Visual Studio. Once this is done APPEND to the list of include directories, library directories and linker input files. DO NOT replace already existing paths.