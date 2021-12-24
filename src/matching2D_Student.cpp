#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        
        if (descriptorType=="DES_HOG"){
            normType = cv::NORM_L2;
            cout<<"switching to L2_NORM for "<< descriptorType<<endl;

        }
            
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if ( descSource.type() != CV_32F ){
            descSource.convertTo(descSource,CV_32F);
            descRef.convertTo(descRef,CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches,2);


        const float ratio_threshold=0.8f;
        for ( size_t i=0; i<knn_matches.size();i++){
            float ratio = knn_matches[i][0].distance / knn_matches[i][1].distance;
            if ( ratio < ratio_threshold )
                matches.push_back(knn_matches[i][0]);

        }
        
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType,float& compute_time)
{
    // select appropriate descriptor
    // BRIEF, ORB, FREAK, AKAZE, SIFT
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        //extractor->compute(img,keypoints,descriptors);
        
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
        //extractor->compute(img,keypoints,descriptors);
        
    }

    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
        //extractor->compute(img,keypoints,descriptors);
        
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
        //extractor->compute(img,keypoints,descriptors);
        
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
        //extractor->compute(img,keypoints,descriptors);
        
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    compute_time = t * 1000 / 1.0;
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis,float& compute_time)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    compute_time = t * 1000 / 1.0;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis,float& compute_time ){

    int blockSize = 2;       //  for every pixel , a block size x block size neighborhood is considered
    int apertureSize = 3;    // odd number for sobel operator
    int minResponse = 100;   // min value for a corner in 8 Bit scaled response matrix
    double k = 0.04;         // Harris Parameter

    
    cv::Mat dest,dest_norm,dest_norm_scaled;
    dest = cv::Mat::zeros(img.size(),CV_32FC1);

    double t = (double)cv::getTickCount();
    cv::cornerHarris(img,dest,blockSize,apertureSize,k,cv::BORDER_DEFAULT);
    cv::normalize(dest, dest_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs( dest_norm, dest_norm_scaled );   

    
    // implement Non maximum Suppression algorithm.
    int row_size = dest_norm.rows;
    int col_size = dest_norm.cols;
    vector<cv::KeyPoint> keyPts;
    double max_overlap = 0.0;

    for(int i = 0; i<row_size; i++){
        for(int j = 0; j<col_size; j++){
            
            unsigned int current_val = dest_norm.at<float>(i,j);
            if (current_val > minResponse){
                
                cv::KeyPoint kp;
                kp.pt = cv::Point2f(j,i);
                kp.response = current_val;
                kp.size = 2 * apertureSize;

                bool is_overlap = false;

                for (cv::KeyPoint kp_it : keyPts){
                    double overlap = cv::KeyPoint::overlap(kp,kp_it);
                    if (overlap > max_overlap){
                        is_overlap = true;
                        if (kp.response > kp_it.response){
                            kp_it = kp;
                        }
                            
                         
                        
                    }


                }
                
                if (is_overlap == false) 
                    keyPts.push_back(kp);
                
                
                    

            }




        }
    }



    



    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    compute_time = t * 1000 / 1.0;
    keypoints = keyPts;
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis,float& compute_time){

    double t;

    if (detectorType.compare("FAST") == 0){

        t = (double)cv::getTickCount();
        cv::Ptr<cv::FastFeatureDetector> fast_detect = cv::FastFeatureDetector::create(30,true);
        fast_detect->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }

    else if (detectorType.compare("BRISK") == 0){

        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "BRISK detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }

    else if (detectorType.compare("ORB") == 0){

        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "ORB detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }

    else if (detectorType.compare("AKAZE") == 0){

        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if (detectorType.compare("SURF") == 0){


        int minHessian=400;
        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create(minHessian);
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "SURF detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if (detectorType.compare("SIFT") == 0){

        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        detector->detect(img,keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "SIFT detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }

    compute_time = t * 1000 / 1.0;
    



}

