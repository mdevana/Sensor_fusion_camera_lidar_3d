
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, string lidarImgFile)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    //cv::namedWindow(windowName, 1);
    //cv::imshow(windowName, topviewImg);
    cv::imwrite(lidarImgFile,topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    
        //cout<<"Total number of Keypoint matches to assign  :"<<kptMatches.size()<<endl;
        vector<cv::DMatch> kpm_for_BB;
        vector<double> euclid_dist;


        for (auto it_kpm = kptMatches.begin(); it_kpm != kptMatches.end() - 1; ++it_kpm){

            cv::KeyPoint kpCurr = kptsCurr.at(it_kpm->trainIdx);
            cv::KeyPoint kpPrev = kptsPrev.at(it_kpm->queryIdx);

            if ( boundingBox.roi.contains(kpCurr.pt) ){
                kpm_for_BB.push_back(*it_kpm);
                euclid_dist.push_back( cv::norm( kpCurr.pt - kpPrev.pt) );
            }


        }

        double mean_dist = std::accumulate(euclid_dist.begin(), euclid_dist.end(), 0.0) / euclid_dist.size();

        
        sort(euclid_dist.begin(), euclid_dist.end());
        int mid_value = floor(euclid_dist.size() / 2);
        double median_dist = ( euclid_dist.size() % 2 == 0 ) ? (euclid_dist[mid_value -1] + euclid_dist[mid_value]) / 2 : euclid_dist[mid_value];

        double first_Q = euclid_dist[mid_value - (int)(mid_value/2)];
        double thrid_Q = euclid_dist[mid_value + (int)(mid_value/2)];
        double range_Q = thrid_Q - first_Q;

        double low_limit = 0;//first_Q - 1.5 * range_Q;
        double high_limit = thrid_Q + 1.5 * range_Q;



        auto it2 = kpm_for_BB.begin();
        for(auto it1 = euclid_dist.begin();it1!=euclid_dist.end();++it1,++it2){
            
            double dist = (*it1);
            if( (dist <= high_limit) && (dist >= low_limit)  ){
                boundingBox.kptMatches.push_back((*it2));
                boundingBox.keypoints.push_back( kptsCurr.at(it2->trainIdx) );
            }
            
        }



        //boundingBox.kptMatches = kpm_for_BB;
        //cout<<"ID :"<<boundingBox.boxID<< " kptmatches :"<<boundingBox.kptMatches.size()<<endl;
    

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    string ratio_type = "MEDIAN"; // "MEAN" or "MEDIAN"
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    /*for(double r:distRatios)
        cout<<"distance ratios ="<<r<<" ";*/

    double distRatio;
    // compute camera-based TTC from distance ratios
    if (ratio_type == "MEAN")
        distRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    else { // Default ratio type is MEDIAN
        sort(distRatios.begin(), distRatios.end());
        int mid_value = floor(distRatios.size() / 2);
        distRatio = ( distRatios.size() % 2 == 0 ) ? (distRatios[mid_value -1] + distRatios[mid_value]) / 2 : distRatios[mid_value];

    }

    double dT = 1 / frameRate;
    TTC = -dT / (1 - distRatio);
    cout<<" distance Ratio ="<<distRatio;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane
    int mid_value;
    double minXPrev = 1e9, minXCurr = 1e9;

    //calculate the median of lidar points as robust way to remove outliers
    sortLidarPointsinX(lidarPointsPrev);
    mid_value = floor(lidarPointsPrev.size() / 2);
    minXPrev = ( lidarPointsPrev.size() % 2 == 0 ) ? (lidarPointsPrev[mid_value -1].x + lidarPointsPrev[mid_value].x) / 2 : lidarPointsPrev[mid_value].x;


    sortLidarPointsinX(lidarPointsCurr);
    mid_value = floor(lidarPointsCurr.size() / 2);
    minXCurr = ( lidarPointsCurr.size() % 2 == 0 ) ? (lidarPointsCurr[mid_value -1].x + lidarPointsCurr[mid_value].x) / 2 : lidarPointsCurr[mid_value].x;


    // find closest distance to Lidar points within ego lane
    /*double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if ( abs(it->y) < laneWidth/2)
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if ( abs(it->y) < laneWidth/2)
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }*/

    // compute TTC from both measurements
    cout<<" min X Prev ="<<minXPrev<<endl;
    cout<<" max X Curr ="<<minXCurr<<endl;
    cout<<"difference ="<<(minXPrev - minXCurr)<<endl;


    TTC = minXCurr * dT / (minXPrev - minXCurr);
}

void sortLidarPointsinX(std::vector<LidarPoint> &l_pts){

    std::sort(l_pts.begin(),l_pts.end(),[](LidarPoint pt1,LidarPoint pt2) {
            return pt1.x < pt2.x;
    });

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{   
    
    vector<BoundingBox> BB_currentFrame = currFrame.boundingBoxes;
    vector<BoundingBox> BB_prevFrame = prevFrame.boundingBoxes;

    multimap<int, int> prev_curr_BB_ids;
    int max_prev_BB_ids= -1;
    int large_prev_id = 0;


    for(auto it2=matches.begin(); it2!=matches.end(); ++it2){
            
            
            cv::KeyPoint kp_matched_curr = currFrame.keypoints.at(it2->trainIdx);
            cv::KeyPoint kp_matched_prev = prevFrame.keypoints.at(it2->queryIdx);

            int prev_BB_id = -1;
            for (BoundingBox BB:BB_prevFrame) {

                if (BB.roi.contains(kp_matched_prev.pt))
                    prev_BB_id = BB.boxID;

            }
            
            int curr_BB_id = -1;
            for (BoundingBox BB:BB_currentFrame) {

                if (BB.roi.contains(kp_matched_curr.pt))
                    curr_BB_id = BB.boxID;

            }

            prev_curr_BB_ids.insert(std::make_pair(curr_BB_id,prev_BB_id));
            large_prev_id = std::max(large_prev_id,prev_BB_id); // pick the largest box id in prev data frame

        }
        
        
        // Loop to find partners for the current frame boundingboxes    
        for (BoundingBox BB:BB_currentFrame) {

            //cout<<"Box ID"<<BB.boxID<<" "<<prev_curr_BB_ids.count(BB.boxID)<<endl;
            vector<int> count_ids(large_prev_id + 1,0); // vector to count the previous box id recurrence in prev data frame

            // Pick occurences for current bounding box as key
            std::pair< std::multimap<int,int>::iterator,std::multimap<int,int>::iterator > ret;
            ret = prev_curr_BB_ids.equal_range(BB.boxID);

            for(std::multimap<int,int>::iterator it_map = ret.first;it_map!=ret.second;it_map++){
                if (it_map->second != -1)
                    count_ids[it_map->second]++;
            }
 
            int sel_id_prev = std::max_element(count_ids.begin(),count_ids.end()) - count_ids.begin();                   
            bbBestMatches.insert({sel_id_prev,BB.boxID});

            
        }

}
