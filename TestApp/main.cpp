#define _CRT_SECURE_NO_DEPRECATE
#include<iostream>
#include<opencv2\core\core.hpp>
#include<opencv2\opencv.hpp>
#include<opencv2\features2d\features2d.hpp>
//Had to compile the nonfree modules seperately because they dont come prebuilt into OpenCV
#include<opencv2\xfeatures2d\nonfree.hpp>
#include<opencv2\ml\ml.hpp>

using namespace cv;
using namespace cv::ml;

int main(int argc, const char* argv[]) {
	//Set Up code
	char* filename = new char[100];
	//Variable to store the input image
	Mat input;
	//Stores the keypoints of the image
	std::vector<KeyPoint> keypoints;
	//Stores the descriptor of the image
	Mat descriptor;
	//Store all such descriptor into one Mat object
	Mat featuresUnclustered;
	//One variable to detect and compute features and their descriptors using SURF algo
	Ptr<cv::xfeatures2d::SurfFeatureDetector> detector = cv::xfeatures2d::SURF::create();
	int j = 1;

	//Construct the initial vocabulary using only 1 image
	sprintf(filename, "C:\\Users\\ANONYMOUS-PC\\Desktop\\CCBD Poject\\images\\FolderA\\a%i.jpg", j);
	input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	std::cout << "Reading:" << filename << std::endl;
	detector->detectAndCompute(input,noArray(),keypoints,descriptor);
	featuresUnclustered.push_back(descriptor);

	//Number of clusters is set to 200
	//Why? Well, you can think of this number as the number of points of similarity
	//As in when a new image comes in, there are 200 categories it has to satisfy to be classified as face or not face
	int numberOfClusters = 200;
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	BOWKMeansTrainer trainer(numberOfClusters,tc,retries,flags);
	//Now the Bag of Words is clustered into a 200 different sets using K-means algorithm
	Mat dictionary = trainer.cluster(featuresUnclustered);

	//Now for the rest of the images the following is done:
	//Get the feature keypoints and their descriptors
	//Set the vocabulary as the Bag Of Words you got from before
	//For the extracted descriptors get the closest Bag Of Words match
	//Label these matches and save both the label and the match
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	Ptr<cv::xfeatures2d::SurfFeatureDetector> extractor = cv::xfeatures2d::SURF::create();
	BOWImgDescriptorExtractor bow(extractor,matcher);
	bow.setVocabulary(dictionary);
	Mat img;
	Mat bowDescrip;
	Mat labels;
	Mat trainingData;
	char* saveFile = new char[100];
	sprintf(saveFile, "classifier.xml");
	
	//First doing the above steps for positive images - hence label is 1
	for (j = 1; j < 151; j++) {
		sprintf(filename, "C:\\Users\\ANONYMOUS-PC\\Desktop\\CCBD Poject\\images\\FolderA\\a%i.jpg", j);
		std::cout << "Reading:" << filename << std::endl;
		img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		extractor->detectAndCompute(img, noArray(), keypoints, descriptor);
		bow.compute2(img,keypoints,bowDescrip);
		trainingData.push_back(bowDescrip);
		labels.push_back(1);
	}
	//Second, doing the same steps for negative images - hence label is 0
	for (j = 1; j < 151; j++) {
		sprintf(filename, "C:\\Users\\ANONYMOUS-PC\\Desktop\\CCBD Poject\\images\\FolderB\\b%i.jpg", j);
		std::cout << "Reading:" << filename << std::endl;
		img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		extractor->detectAndCompute(img, noArray(), keypoints, descriptor);
		bow.compute2(img, keypoints, bowDescrip);
		trainingData.push_back(bowDescrip);
		labels.push_back(0);
	}

	//Now the training data and labels need to be converted into Ptr<TrainData> format
	//This is then used to train the SVM
	Ptr<TrainData> trainDataFormatted = TrainData::create(trainingData,SampleTypes::ROW_SAMPLE,labels);
	Ptr<SVM> svm = SVM::create();
	//Used the method 'train' before
	//Had to manually set the parameters for that and somehow that didnt work
	//Instead used TrainAuto which automatically sets the default parameters for the training - kernel,gamma etc etc.
	svm->trainAuto(trainDataFormatted);
	//Save the classifier file after the training has finished
	svm->save(saveFile);
	//Now start the prediction for images
	//Repeat the steps as before 
	//Get the Bag of Words descriptor for the vocabulary set
	//Not instead of training we will predict, which returns the label of the class that image belongs to
	int ctr = 0;
	for (j = 151; j < 201; j++) {
		sprintf(filename, "C:\\Users\\ANONYMOUS-PC\\Desktop\\CCBD Poject\\images\\FolderA\\a%i.jpg", j);
		Mat sampleMat = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		extractor->detectAndCompute(sampleMat, noArray(), keypoints, descriptor);
		bow.compute2(sampleMat, keypoints, bowDescrip);
		float response = svm->predict(bowDescrip);
		std::cout << "Filename:" << filename << std::endl;
		std::cout << "Result:" << response << std::endl;
		if (response == 1) {	
			ctr++;
		}
	}
	std::cout << "Folder A:" << std::endl;
	std::cout << "Number of correct predictions:" << ctr << std::endl;
	std::cout << "Number of incorrect predictions:" << 50 - ctr << std::endl;
	getchar();
	ctr = 0;

	for (j = 151; j < 201; j++) {
		sprintf(filename, "C:\\Users\\ANONYMOUS-PC\\Desktop\\CCBD Poject\\images\\FolderB\\b%i.jpg", j);
		Mat sampleMat = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		extractor->detectAndCompute(sampleMat, noArray(), keypoints, descriptor);
		bow.compute2(sampleMat, keypoints, bowDescrip);
		float response = svm->predict(bowDescrip);
		std::cout << "Filename:" << filename << std::endl;
		std::cout << "Result:" << response << std::endl;
		if (response == 0) {
			ctr++;
		}
	}
	std::cout << "Folder B:" << std::endl;
	std::cout << "Number of correct predictions:" << ctr << std::endl;
	std::cout << "Number of incorrect predictions:" << 50 - ctr << std::endl;

	getchar();
	return 0;
}