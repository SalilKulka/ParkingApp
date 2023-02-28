#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

#include "NumCpp.hpp"

#include "opencv2/core/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#define TOLERANCE 0.01

const int HEIGHT = 400;
const int WIDTH = 500;

//example of interfacing NumCpp with OpenCV - https://dpilger26.github.io/NumCpp/doxygen/html/_interface_with_open_c_v_8cpp-example.html

//using cv::Point

cv::Point selectFarthestPoint(cv::Point &p1, std::vector<cv::Point> &ptList){
    //returns either of p2 or p3 depending on which point is farthest away from p1 (mounting point)
    double maxDistance = 0;
    double d1 = 0;
    cv::Point p = cv::Point(0,0);
    int len = ptList.size();
    for(int i = 0; i<len; i++){
        d1 = abs(sqrt(pow(p1.x-ptList[i].x,2)+pow(p1.y-ptList[i].y,2)));
        if(d1>maxDistance){       
            p.x = ptList[i].x;
            p.y = ptList[i].y;
            maxDistance = d1;
        }
    }
    return p;
}

void getImgOutlineBorderIntersection(cv::Mat &cutQuadBorder, cv::Mat &retImg){
    cv::Mat imgOutline = cv::Mat::zeros(cutQuadBorder.size(), CV_8UC3);
    std::vector<cv::Point> imgBorderPts = {cv::Point(0,0), cv::Point(WIDTH, 0), cv::Point(WIDTH, HEIGHT), cv::Point(0, HEIGHT)};
    cv::polylines(imgOutline, imgBorderPts, true, cv::Scalar(255));
    cv::bitwise_and(imgOutline, cutQuadBorder, retImg);

}

void getRoadsnParkings(cv::Mat &img, cv::Mat &blue_mask){




    cv::inRange(img, cv::Scalar(255,0,0), cv::Scalar(255,0,0), blue_mask);


}


void cluster2Point(cv::Mat &clusterImg, cv::Mat &img, std::vector<cv::Point>* Points){

    int cx,cy;
    cv::Point c = cv::Point(0,0);
    cv::Mat imageCopy = cv::Mat::zeros(clusterImg.size(), CV_8UC3);

    cv::Mat white_mask; 
    cv::Mat combined_mask;

    std::vector<std::vector<cv::Point> > cnts;
    

    if(clusterImg.channels()==1)    //if grayscale
        cv::inRange(clusterImg, cv::Scalar(255), cv::Scalar(255), white_mask);
    else if(clusterImg.channels()==3)   //if color image (3 layers)
        cv::inRange(clusterImg, cv::Scalar(255,255,255), cv::Scalar(255,255,255), white_mask);


    cv::Mat kernel = cv::Mat::ones(cv::Size(3, 3), CV_8UC1);
    cv::morphologyEx(white_mask, combined_mask, cv::MORPH_DILATE, kernel);

    // cv::imshow("Display window 3", combined_mask);
    // k = cv::waitKey(0); // Wait for a keystroke in the window

    cv::findContours(combined_mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for(int i = 0; i<cnts.size(); i++){
        cv::Moments M = cv::moments(cnts[i]);
        if (M.m00 != 0){
            cx = int(M.m10/M.m00);
            cy = int(M.m01/M.m00);
        }
        Points->push_back(cv::Point(cx, cy));
        // # print(f"center - {cx},{cy}")
        cv::circle(img, cv::Point(cx, cy), 1, cv::Scalar(255, 255, 255), -1);
    }

}

void cluster2Point_noDraw(cv::Mat &clusterImg, std::vector<cv::Point>* Points){

    int cx,cy;
    cv::Point c = cv::Point(0,0);
    cv::Mat imageCopy = cv::Mat::zeros(clusterImg.size(), CV_8UC3);

    cv::Mat white_mask; 
    cv::Mat combined_mask;

    std::vector<std::vector<cv::Point> > cnts;
    

    if(clusterImg.channels()==1)    //if grayscale
        cv::inRange(clusterImg, cv::Scalar(255), cv::Scalar(255), white_mask);
    else if(clusterImg.channels()==3)   //if color image (3 layers)
        cv::inRange(clusterImg, cv::Scalar(255,255,255), cv::Scalar(255,255,255), white_mask);


    cv::Mat kernel = cv::Mat::ones(cv::Size(3, 3), CV_8UC1);
    cv::morphologyEx(white_mask, combined_mask, cv::MORPH_DILATE, kernel);

    // cv::imshow("Display window 3", combined_mask);
    // k = cv::waitKey(0); // Wait for a keystroke in the window

    cv::findContours(combined_mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for(int i = 0; i<cnts.size(); i++){
        cv::Moments M = cv::moments(cnts[i]);
        if (M.m00 != 0){
            cx = int(M.m10/M.m00);
            cy = int(M.m01/M.m00);
        }
        Points->push_back(cv::Point(cx, cy));
        // # print(f"center - {cx},{cy}")
    }

}

void getMountingPoints(cv::Mat &imageCopy) {


    double maxVal = 0;

    // Upper and lower color limit customized for snazzy maps
    cv::Scalar low_red = cv::Scalar(55, 55, 255);

    // create masks
    cv::Mat red_mask;
    cv::inRange(imageCopy, low_red, low_red, red_mask);

    // combine masks
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat combined_mask;
    cv::dilate(red_mask, combined_mask, kernel);

    // setting to 32-bit floating point
    cv::Mat operatedImage;
    combined_mask.convertTo(operatedImage, CV_32F);

    // apply the cv::cornerHarris method
    // to detect the corners with appropriate values as input parameters
    cv::Mat dest;
    cv::cornerHarris(operatedImage, dest, 2, 3, 0.04);

    // Results are marked through the dilated corners
    cv::dilate(dest, dest, kernel);

    // draw on the output image
    // cv::Mat whitePixels = cv::Mat::zeros(imageCopy.size(), imageCopy.type());
    // whitePixels.setTo(cv::Scalar(255, 255, 255));
    cv::minMaxLoc(dest, NULL, &maxVal, NULL, NULL);
    imageCopy.setTo(cv::Scalar(255, 255, 255), dest > 0.01 * maxVal);

}
// Make the corners into one point

    // innitates the the single point corner arrays

bool rangeCheck(float checkAgainst, float val){
    if(val>=checkAgainst*(1-TOLERANCE) && val<=checkAgainst*(1-TOLERANCE))
        return true;
    return false;
}
    



void pointGen(cv::Point source, float m, float l, std::vector<cv::Point> &retArray){  //retArray is an array of integer pointers

    if(m==0){
        retArray[0].x = int(source.x+l);    //a.x corresponding to python code
        retArray[1].y = int(source.y);  //a.y corresponding to python code

        retArray[1].x = int(source.x-l);    //b.x corresponding to python code
        retArray[1].y = int(source.y);  //b.y corresponding to python code
    }

    else if(!isfinite(m)){
        retArray[0].x = int(source.x);
        retArray[1].y = int(source.y + l);

        retArray[1].x = int(source.x);
        retArray[1].y = int(source.y - l);
    }
    else{
        float dx = (l / sqrt(1 + (m * m)));
        float dy = (m * dx);
        retArray[0].x = int(source.x + dx);
        retArray[1].y = int(source.y + dy);
        retArray[1].x = int(source.x - dx);
        retArray[1].y = int(source.y - dy);
    }

}

void drawLine(float slope, cv::Point point, cv::Mat &returnImageMatrix){
    
    cv::Point p = cv::Point(0,0);
    cv::Point q = cv::Point(0,0);

    if(!isfinite(slope))
    {   p.x = point.x;
        p.y = HEIGHT;

        q.x = point.x;
        q.y = 0;
        }
    else
        {   p.x = WIDTH;
        p.y = int(slope*WIDTH + (point.y - slope*point.x));

        q.x = 0;
        q.y = int(point.y - slope*point.x);
        }

    
    cv::line(returnImageMatrix, p, q, cv::Scalar(255), 1, 8);
    //removed try except over here - add again if necessary
      
}

cv::Point cornerInclude(cv::Point p1, cv::Point p2, cv::Point mountingPoint, cv::Mat &cutQuadBorderPtsOnimgOutline) {
    int h = cutQuadBorderPtsOnimgOutline.rows;
    int w = cutQuadBorderPtsOnimgOutline.cols;
    cv::Mat intersection;
    std::vector<cv::Point> points;

    cv::Mat blank = cv::Mat::zeros(cutQuadBorderPtsOnimgOutline.size(), CV_8UC1);

    // Check if p1 and p2 lie on different vertical borders
    if ((rangeCheck(p1.y, h) && !rangeCheck(p2.y, h)) || (rangeCheck(p2.y, h) && !rangeCheck(p1.y, h))) {
        cv::line(blank, cv::Point(0, h), cv::Point(w, h), cv::Scalar(255), 5);
        
        cv::bitwise_and(blank, cutQuadBorderPtsOnimgOutline, intersection);
        
        cv::findNonZero(intersection, points);
        return selectFarthestPoint(mountingPoint, points);
    }
    // Check if p1 and p2 lie on different horizontal borders
    else if ((rangeCheck(p1.y, 0) && !rangeCheck(p2.y, 0)) || (rangeCheck(p2.y, 0) && !rangeCheck(p1.y, 0))) {
        cv::line(blank, cv::Point(0, 0), cv::Point(w, 0), cv::Scalar(255), 5);
        
        cv::bitwise_and(blank, cutQuadBorderPtsOnimgOutline, intersection);
        
        cv::findNonZero(intersection, points);
        return selectFarthestPoint(mountingPoint, points);
    }
    // Check if p1 and p2 lie on different right-side borders
    else if ((rangeCheck(p1.x, w) && !rangeCheck(p2.x, w)) || (rangeCheck(p2.x, w) && !rangeCheck(p1.x, w))) {
        cv::line(blank, cv::Point(w, 0), cv::Point(w, h), cv::Scalar(255), 5);
        
        cv::bitwise_and(blank, cutQuadBorderPtsOnimgOutline, intersection);
        
        cv::findNonZero(intersection, points);
        return selectFarthestPoint(mountingPoint, points);
    }
    // Check if p1 and p2 lie on different left-side borders
    else if ((rangeCheck(p1.x, 0) && !rangeCheck(p2.x, 0)) || (rangeCheck(p2.x, 0) && !rangeCheck(p1.x, 0))) {
        cv::line(blank, cv::Point(0, 0), cv::Point(0, h), cv::Scalar(255), 5);
        
        cv::bitwise_and(blank, cutQuadBorderPtsOnimgOutline, intersection);
        
        cv::findNonZero(intersection, points);
        return selectFarthestPoint(mountingPoint, points);
    }

    return cv::Point(-1, -1); // If none of the above conditions are met, return an invalid point.
}

// Function to generate a mask and border for a quadrilateral given its vertices
// Arguments:
// - pt: the vertices of the quadrilateral
// Returns:
// - cutQuadMask: a binary mask of the quadrilateral
// - cutQuadBorder: a binary image of the quadrilateral border
void genQuadImages(std::vector<cv::Point> pt, cv::Mat &cutQuadMask, cv::Mat &cutQuadBorder) {

    cv::fillPoly(cutQuadMask, pt, cv::Scalar(255, 255, 255));

    // Find the contours of the filled quadrilateral
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(cutQuadMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Approximate the contours with a line segment, and draw the line segment on the border image
    for (auto& c : contours) {
        for (double eps : nc::linspace<double>(0.001, 0.01, 10)) { // Iterate over 10 values between 0.001 and 0.01
            // Approximate the contour with line segments
            double peri = cv::arcLength(c, true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(c, approx, eps * peri, true);

            // Draw the line segment on the border image
            cv::polylines(cutQuadBorder, std::vector<std::vector<cv::Point> >{approx}, true, cv::Scalar(255, 255, 255), 1);
        }
    }

}

void getRoadCoverageMask(std::vector<std::vector<cv::Point>> selected_edge_list, cv::Mat cutQuadBorder, cv::Point mountingPoint, cv::Mat &totalMask){
    cv::Mat outLineImgIntersection;
    getImgOutlineBorderIntersection(cutQuadBorder, outLineImgIntersection);

    for(auto& selected_edge_contour : selected_edge_list){
        //mask related to a single contour, to which individual line blocking masks will be OR'ed to
        cv::Mat contourMask = cv::Mat::zeros(cutQuadBorder.size(), CV_8UC1);

        cv::Mat oneContourPic = cv::Mat::zeros(cutQuadBorder.size(), CV_8UC3);
        std::vector<std::vector<cv::Point>> contours{selected_edge_contour};
        cv::drawContours(oneContourPic, contours, 0, cv::Scalar(255,255,255), 1); //draws only selected contour for the loop
        cv::cvtColor(oneContourPic, oneContourPic, cv::COLOR_BGR2GRAY);   //convert drawn picture into grayscale

        cv::Canny(oneContourPic, oneContourPic, 50, 150, 3);

        // Apply HoughLinesP method to 
        // to directly obtain line end points
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(
                    oneContourPic, // Input edge image
                    lines, // Output lines
                    1, // Distance resolution in pixels
                    CV_PI/180, // Angle resolution in radians
                    11, // Min number of votes for valid line. Use lesser number of votes for smaller lines
                    5, // Min allowed length of line
                    10 // Max allowed gap between line for joining them
                    );

        //same line is appearing in the pic twice, perhaps increase threshold (number of votes/points)
        if(lines.empty()){
            continue;
        }

        cv::Mat lineMask = cv::Mat::zeros(cutQuadBorder.size(), CV_8UC1);
        for(auto& points : lines){
            cv::Mat joiningLine_1 = cv::Mat::zeros(cutQuadBorder.size(), CV_8UC1);
            cv::Mat joiningLine_2 = cv::Mat::zeros(cutQuadBorder.size(), CV_8UC1);

            cv::Point finalPointCoords_1;
            cv::Point finalPointCoords_2;

            // Extracted points nested in the list
            int x1 = points[0];
            int y1 = points[1];
            int x2 = points[2];
            int y2 = points[3];

            // Step 4: For each line AND it with the interior corner points ??

            cv::Point interiorPointCoords_1 = cv::Point(x1, y1);
            cv::Point interiorPointCoords_2 = cv::Point(x2, y2);

            // Step 5: Isolate the points and their coordinates that intersect with the contour line
            // Step 6: Get the slope from the mounting point to those interior corners
            // Step 7: Drop a line from the interior corners to edge of image on a quad mask so you dont have to find intersection point with further edge
            if(mountingPoint.x-interiorPointCoords_1.x==0){
                float slope1 = (mountingPoint.y - interiorPointCoords_1.y)/(mountingPoint.x - interiorPointCoords_1.x);
                drawLine(slope1, interiorPointCoords_1, joiningLine_1);
            }
            else{
                drawLine(INFINITY, interiorPointCoords_1, joiningLine_1);
            }

            if(mountingPoint.x-interiorPointCoords_2.x==0){
                float slope2 = (mountingPoint.y - interiorPointCoords_2.y)/(mountingPoint.x - interiorPointCoords_2.x);
                drawLine(slope2, interiorPointCoords_2, joiningLine_2);
            }

            cv::Mat Point_1;
            cv::bitwise_and(cutQuadBorder, joiningLine_1, Point_1);
            std::vector<cv::Point> PointCoords_1;
            cluster2Point_noDraw(Point_1, &PointCoords_1);    //NULL might be potential cause of error here. Replace if necessary

            if(PointCoords_1.size()==0)  //weird case that you can't do anything about. Realistically this shouldn't exist, but it does
                continue;
            else if(PointCoords_1.size()==1)
                finalPointCoords_1 = PointCoords_1[0];
            else if(PointCoords_1.size()>=2){
                finalPointCoords_1 = selectFarthestPoint(mountingPoint, PointCoords_1);
            }

            cv::Mat Point_2;
            cv::bitwise_and(cutQuadBorder, joiningLine_2, Point_2);
            std::vector<cv::Point> PointCoords_2;
            cluster2Point_noDraw(Point_2, &PointCoords_2);    //NULL might be potential cause of error here. Replace if necessary

            if(PointCoords_2.size()==0)  //weird case that you can't do anything about. Realistically this shouldn't exist, but it does
                continue;
            else if(PointCoords_2.size()==1)
                finalPointCoords_2 = PointCoords_2[0];
            else if(PointCoords_2.size()>=2){
                finalPointCoords_2 = selectFarthestPoint(mountingPoint, PointCoords_2);
            }
            std::vector<cv::Point> pointSet = {interiorPointCoords_1, interiorPointCoords_2, finalPointCoords_2, finalPointCoords_1};
            
            cv::Point inclusionCorner = cornerInclude(finalPointCoords_1, finalPointCoords_2, mountingPoint, outLineImgIntersection);
            if(inclusionCorner.x!=-1)
                pointSet = {interiorPointCoords_1, interiorPointCoords_2, finalPointCoords_2, inclusionCorner, finalPointCoords_1};


            // Step 8: Fill the quadrilateral made by the line contour, the lines from interior corners or the view quad side wall and the view quad further wall
            // There is a concern that the points are listed in not the correct order...

            cv::fillPoly(lineMask, pointSet,cv::Scalar(255));
        }

            // Step 9: OR all the quadrilateral generated inside the loop on the view quad mask
            cv::bitwise_or(contourMask, lineMask, contourMask);
        cv::bitwise_or(totalMask, contourMask, totalMask);
    cv::bitwise_not(totalMask, totalMask);
    }
}

int main(){
    std::vector<cv::Point> test = {cv::Point(0,0),cv::Point(0,0)};
    cv::Point testSrc = cv::Point(5,5);
    pointGen(testSrc, 6, 10, test);
    for(int x = 0; x<2; x++){

        std::cout<<test[x].x;
        std::cout<<" ";
        std::cout<<test[x].y;
        
        std::cout<<"\n";
    }

    cv::Mat img = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));

    drawLine(20, testSrc, img);
    // cv::namedWindow("Display Window");

    cv::imshow("Display window", img);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        cv::imwrite("starry_night.png", img);
    }

    std::string image_path = cv::samples::findFile("C:\\Users\\athyn\\Desktop\\Projects\\Tarq\\Parking\\ParkingApp\\Images\\sample3.png");
    cv::Mat img2 = cv::imread(image_path, cv::IMREAD_COLOR);

    getRoadsnParkings(img2, img);


    cv::imshow("Display window 2", img);
    k = cv::waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        cv::imwrite("starry_night.png", img);
    }

    test = {cv::Point(200,100), cv::Point(300,400), cv::Point(500,400), cv::Point(600,200)};
    cv::Mat cutQuadMask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::Mat cutQuadBorders = cv::Mat::zeros(img.size(), CV_8UC1);
    genQuadImages(test, cutQuadMask, cutQuadBorders);
    
    cv::imshow("cut quad borders", cutQuadBorders);
    cv::imshow("cut quad mask", cutQuadMask);
    k = cv::waitKey(0); // Wait for a keystroke in the window

    // testing code for getMountingPoints
    // img2 = cv::imread(image_path, cv::IMREAD_COLOR);
    // img = cv::imread(image_path, cv::IMREAD_COLOR);
    // getMountingPoints(img);
    // std::vector<cv::Point> collector;
    // cluster2Point(img, img2, &collector);
    // cv::imshow("Display window mounting points", img2);
    // k = cv::waitKey(0); // Wait for a keystroke in the window


    // testing code for cluster2point
    // img = cv::Mat::zeros(img2.size(), CV_8UC1);
    // img2 = cv::Mat::zeros(img2.size(), CV_8UC3);
    // cv::circle(img, testSrc, 8, cv::Scalar(255), cv::FILLED);
    // testSrc.x = 200;
    // testSrc.y = 100;
    // cv::circle(img, testSrc, 8, cv::Scalar(255), cv::FILLED);


    // cluster2Point(img, img2, &collector);
    // std::cout<<collector.size();
    // for(int i = 0; i<collector.size(); i++){
    //     std::cout<<collector[i].x;
    //     std::cout<<" ";
    //     std::cout<<collector[i].y<<std::endl;
    // }

    // cv::imshow("Display window", img);
    // cv::imshow("Display window 2", img2);
    // k = cv::waitKey(0); // Wait for a keystroke in the window

    // return 0;
}
