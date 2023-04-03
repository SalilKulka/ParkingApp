#include <iostream>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <cmath>
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

    // if(m==0){
    //     retArray[0].x = int(source.x+l);    //a.x corresponding to python code
    //     retArray[0].y = int(source.y);  //a.y corresponding to python code

    //     retArray[1].x = int(source.x-l);    //b.x corresponding to python code
    //     retArray[1].y = int(source.y);  //b.y corresponding to python code
    // }

    if(!isfinite(m)){
        retArray[0].x = int(source.x);
        retArray[0].y = int(source.y + l);

        retArray[1].x = int(source.x);
        retArray[1].y = int(source.y - l);
    }
    else{
        float dx = (l / sqrt(1 + (m * m)));
        float dy = (m * dx);
        retArray[0].x = int(source.x + dx);
        retArray[0].y = int(source.y + dy);
        retArray[1].x = int(source.x - dx);
        retArray[1].y = int(source.y - dy);
    }

}

// calculates the Euclidean distance between points a and b
double distance(cv::Point a, cv::Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
}

// checks if point c lies on the line segment ab
bool is_between(cv::Point a, cv::Point c, cv::Point b) {
    double softCheck = distance(a, c) + distance(c, b) - distance(a, b);
    return std::abs(softCheck) <= 0.5;
}

void drawLineOld(float slope, cv::Point point, cv::Mat &returnImageMatrix){
    
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

void drawLine(cv::Point mountingPt, cv::Point interiorPoint_1, cv::Point interiorPoint_2, int y_lim, int x_lim, cv::Point &p, cv::Point &q) {


    // if the line is vertical
    if (mountingPt.x == interiorPoint_1.x) {

        // set the x-coordinate of p to the x-coordinate of the line
        p.x = interiorPoint_1.x;
        // set the y-coordinate of p to the bottom of the image
        p.y = y_lim;

        // if mounting point is not between the two points on the line
        if (!is_between(mountingPt, interiorPoint_1, p)) {
            // set the y-coordinate of p to the top of the image
            p.x = interiorPoint_1.x;
            p.y = 0;
        }

    // if the line is not vertical
    } else {
        // calculate the slope of the line between the mounting point and the first point
        float slope1 = (mountingPt.y - interiorPoint_1.y) / (mountingPt.x - interiorPoint_1.x);

        // set the x-coordinate of p to the right edge of the image
        p.x = x_lim;
        // set the y-coordinate of p based on the slope of the line and the y-intercept of the line
        p.y = slope1 * x_lim + (interiorPoint_1.y - slope1 * interiorPoint_1.x);

        // if mounting point is not between the two points on the line
        if (!is_between(mountingPt, interiorPoint_1, p)) {
            // set the x-coordinate of p to the left edge of the image
            p.x = 0;
            // set the y-coordinate of p based on the slope of the line and the y-intercept of the line
            p.y = (interiorPoint_1.y - slope1 * interiorPoint_1.x);
        }
    }

    // if the line is vertical
    if (mountingPt.x == interiorPoint_2.x) {

        // set the x-coordinate of q to the x-coordinate of the line
        q.x = interiorPoint_2.x;
        // set the y-coordinate of q to the bottom of the image
        q.y = y_lim;

        // if mounting point is not between the two points on the line
        if (!is_between(mountingPt, interiorPoint_2, q)) {
            // set the y-coordinate of q to the top of the image
            q.x = interiorPoint_2.x;
            q.y = 0;
        }

    // if the line is not vertical
    } else {
        // calculate the slope of the line between the mounting point and the second point
        float slope2 = (mountingPt.y - interiorPoint_2.y) / (mountingPt.x - interiorPoint_2.x);

        // set the x-coordinate of q to the right edge of the image
        q.x = x_lim;
        // set the y-coordinate of q based on the slope of the line and the y-intercept of the line
        q.y = slope2 * x_lim + (interiorPoint_2.y - slope2 * interiorPoint_2.x);
        if(!is_between(mountingPt, interiorPoint_2, q)){
            q.x = 0;
            q.y = (interiorPoint_2.y - slope2*interiorPoint_2.x);
        }
    }

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
    std::vector<std::vector<cv::Point>> contours;
    // // std::cout<<"HERE\n\n\n\n";
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

void getRoadCoverageMaskOld(std::vector<std::vector<cv::Point>> selected_edge_list, cv::Mat cutQuadBorder, cv::Point mountingPoint, cv::Mat &totalMask){
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
                drawLineOld(slope1, interiorPointCoords_1, joiningLine_1);
            }
            else{
                drawLineOld(INFINITY, interiorPointCoords_1, joiningLine_1);
            }

            if(mountingPoint.x-interiorPointCoords_2.x==0){
                float slope2 = (mountingPoint.y - interiorPointCoords_2.y)/(mountingPoint.x - interiorPointCoords_2.x);
                drawLineOld(slope2, interiorPointCoords_2, joiningLine_2);
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

void getRoadCoverageMask(std::vector<std::vector<cv::Point>> selected_edge_list, cv::Point mountingPoint, int yLim, int xLim, cv::Mat &totalMask) {
    for (auto selected_edge_contour : selected_edge_list) {

        // mask related to a single contour, to which individual line blocking masks will be OR'ed to 
        cv::Mat oneContourPic = cv::Mat::zeros(totalMask.size(), CV_8UC1);

        cv::drawContours(oneContourPic, std::vector<std::vector<cv::Point>>{selected_edge_contour}, -1, cv::Scalar(255), 1); //draws only selected contour for the loop

        cv::Canny(oneContourPic, oneContourPic, 50, 150, 3);

        // Apply HoughLinesP method to to directly obtain line end points
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(oneContourPic, lines, 1, CV_PI/180, 11, 5, 10);

        // same line is appearing in the pic twice, perhaps increase threshold (number of votes/points)
        if(lines.empty()) {
            continue;
        }

        for (auto points : lines) {

            // Step 4: For each line AND it with the interior corner points ??

            cv::Point interiorPointCoords_1(points[0], points[1]);
            cv::Point interiorPointCoords_2(points[2], points[3]);

            // Step 5: Isolate the points and their coordinates that intersect with the contour line
            // Step 6: Get the slope from the mounting point to those interior corners
            // Step 7: Drop a line from the interior corners to edge of image on a quad mask so you dont have to find intersection point with further edge
            
            cv::Point brdrPoint_1, brdrPoint_2;
            drawLine(mountingPoint, interiorPointCoords_1, interiorPointCoords_2, yLim, xLim, brdrPoint_1, brdrPoint_2);

            std::vector<cv::Point> exPoints;
            if (brdrPoint_1.x == 0 && brdrPoint_2.y == 0 || brdrPoint_1.y == 0 && brdrPoint_2.x == 0) {
                exPoints = {interiorPointCoords_1, brdrPoint_1, {0,0}, brdrPoint_2, interiorPointCoords_2};
            }
            else if (brdrPoint_1.x == xLim && brdrPoint_2.y == 0 || brdrPoint_1.y == 0 && brdrPoint_2.x == xLim) {
                exPoints = {interiorPointCoords_1, brdrPoint_1, {xLim,0}, brdrPoint_2, interiorPointCoords_2};
            }
            else if (brdrPoint_1.x == 0 && brdrPoint_2.y == yLim || brdrPoint_1.y == yLim && brdrPoint_2.x == 0) {
                exPoints = {interiorPointCoords_1, brdrPoint_1, {0,yLim}, brdrPoint_2, interiorPointCoords_2};
            }else
                exPoints = {interiorPointCoords_1, brdrPoint_1, brdrPoint_2, interiorPointCoords_2};
            
            
            // fill poly for excluded region
            cv::fillPoly(totalMask, exPoints,cv::Scalar(255));
        }
    }
    cv::bitwise_not(totalMask, totalMask);
    // demo = img_gray.copy()
    // demo = cv2.bitwise_and(demo, totalMask)
    // cv2.circle(demo, mountingPoint, 2, 255, -1)
    // demo = cv2.bitwise_and(cutQuadBorder, demo)
    // cv2.imshow('this', demo)
    // cv2.waitKey(0)
    // cv2.destroyAllWindows()
}


void getBorderContour(cv::Mat img, cv::Mat &blank, cv::Mat &combined_mask) {
    //using new bgr values for the new image in low red.
    cv::Scalar red = cv::Scalar(55, 55, 255);

    // create masks
    cv::Mat red_mask;
    cv::inRange(img, red, red, red_mask);

    // combine masks
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::dilate(red_mask, combined_mask, kernel);



    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(combined_mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < cnts.size(); i++) {
        // cv::drawContours(img, cnts, i, cv::Scalar(255, 0, 255), 1);
        double area = cv::contourArea(cnts[i]);
        if (area > 200) {
            for (double eps = 0.001; eps <= 0.01; eps += 0.001) {
                // approximate the contour
                double peri = cv::arcLength(cnts[i], true);
                std::vector<cv::Point> approx;
                cv::approxPolyDP(cnts[i], approx, eps * peri, true);
                // draw the approximated contour on the image
                cv::drawContours(blank, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(255, 255, 255), 1);
                // cv::drawContours(blank, cnts, i, cv::Scalar(255, 255, 255), 1);
            }
        }
    }

    // cv::imshow("image", img);
    // cv::waitKey(0);
}

int main(){
    // std::vector<cv::Point> test = {cv::Point(0,0),cv::Point(0,0)};
    // cv::Point testSrc = cv::Point(5,5);
    // pointGen(testSrc, 6, 10, test);
    // for(int x = 0; x<2; x++){

    //     // std::cout<<test[x].x;
    //     // std::cout<<" ";
    //     // std::cout<<test[x].y;
        
    //     // std::cout<<"\n";
    // }

    // cv::Mat img = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));

    // drawLineOld(20, testSrc, img);
    // // cv::namedWindow("Display Window");

    // cv::imshow("Display window", img);
    // int k = cv::waitKey(0); // Wait for a keystroke in the window
    // if(k == 's')
    // {
    //     cv::imwrite("starry_night.png", img);
    // }

    // std::string image_path = cv::samples::findFile("C:\\Users\\athyn\\Desktop\\Projects\\Tarq\\Parking\\ParkingApp\\Images\\sample3.png");
    // cv::Mat img2 = cv::imread(image_path, cv::IMREAD_COLOR);

    // getRoadsnParkings(img2, img);


    // cv::imshow("Display window 2", img);
    // k = cv::waitKey(0); // Wait for a keystroke in the window
    // if(k == 's')
    // {
    //     cv::imwrite("starry_night.png", img);
    // }

    // test = {cv::Point(200,100), cv::Point(300,400), cv::Point(500,400), cv::Point(600,200)};
    // cv::Mat cutQuadMask = cv::Mat::zeros(img.size(), CV_8UC1);
    // cv::Mat cutQuadBorders = cv::Mat::zeros(img.size(), CV_8UC1);
    // genQuadImages(test, cutQuadMask, cutQuadBorders);
    
    // cv::imshow("cut quad borders", cutQuadBorders);
    // cv::imshow("cut quad mask", cutQuadMask);
    // k = cv::waitKey(0); // Wait for a keystroke in the window

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
    // // std::cout<<collector.size();
    // for(int i = 0; i<collector.size(); i++){
    //     // std::cout<<collector[i].x;
    //     // std::cout<<" ";
    //     // std::cout<<collector[i].y<<std::endl;
    // }

    // cv::imshow("Display window", img);


    // return 0;
    std::string image_path = cv::samples::findFile("C:\\Users\\athyn\\Desktop\\Projects\\Tarq\\Parking\\ParkingApp-Optimized\\ParkingApp\\Images\\sample3.png");
    cv::Mat origImg = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat road;
    cv::Mat bldg_brdr = cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0));
    cv::Mat bldg_mask = cv::Mat(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0));
    cv::Mat bldg_brdr_gray = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    cv::Mat cutQuadBrdr = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    cv::Mat cutQuadMask = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    // cv::Mat CameraCoverage2 = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    cv::Mat roadCoveredMask = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    cv::Mat imgCopy = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    cv::Mat maxCameraRoadCoverage = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    cv::Mat Check_step1 = cv::Mat::zeros(origImg.rows, origImg.cols, CV_8UC1);
    cv::Mat Check_step2 = cv::Mat::zeros(origImg.rows, origImg.cols, CV_8UC1);

    std::vector<cv::Point> pointCollector = {cv::Point(0,0), cv::Point(0,0)};
    cv::Point fp1, fp2, fp3;
    int k, uncalls = 0, calls = 0, exits = 0;
    int yLim = origImg.rows;
    int xLim = origImg.cols;

    float maxArea = 0;
    float theta = 66.75*M_PI;    //diagonal angle FOV of camera (GIVEN!!)
    float phi = 2*atan(0.8*tan(theta/2));  //angle of view larger side of camera resolution (4 in 4:3)
    float omega = 2*atan(0.6*tan(theta/2));     //angle of view larger side of camera resolution (3 in 4:3)

    getRoadsnParkings(origImg, road);
    cv::imshow("only road", road);
    k = cv::waitKey(0); // Wait for a keystroke in the window
    int scale = 162;
    int scaleConst = 20;
    float heightPix[31];
    std::vector<cv::Point> mountingPointList;
    for(int i = 30; i<61; i++){
        heightPix[i-30] = (i*scale)/20;
    }

    cv::Mat mountingPointCluster = cv::imread(image_path, cv::IMREAD_COLOR);
    getMountingPoints(mountingPointCluster);
    cluster2Point_noDraw(mountingPointCluster, &mountingPointList);
    // // std::cout<<mountingPointList;
    getBorderContour(origImg, bldg_brdr, bldg_mask);
    cv::imshow("bldg Mask", bldg_mask);
    k = cv::waitKey(0);
    cv::destroyAllWindows();
    // cv::imshow("Bldg Brdr", bldg_brdr);
    k = cv::waitKey(0); // Wait for a keystroke in the window
    cv::cvtColor(bldg_brdr, bldg_brdr_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(origImg, imgCopy, cv::COLOR_BGR2GRAY);
    cv::Mat roadCopy = road.clone();
    int time_before_loop = time(NULL);
    float ALPHA = 60*M_PI/180;


    for (const cv::Point& mountingPoint : mountingPointList) {
        for (int beta = 0; beta < 360; beta += 10) {
            uncalls++;
            // beta edge cases
            if (beta == 180 || beta == 0 || beta == 360) {
                continue;
            }

            // loop vars
            double BETA = beta * M_PI / 180;
            cv::Mat cameraRoadCoverage(origImg.rows, origImg.cols, CV_8UC1, cv::Scalar(0));

            // distances of closer and further edges from mounting point
            // closer_dist = heightPix[20]*math.tan(ALPHA - (phi/2))
            double further_dist = heightPix[20] * std::tan(ALPHA + (phi/2));

            // slope of horizontal plane camera angle
            double slope_beta = std::tan(BETA);

            // midpoints of closer and further edges
            cv::Point further_midPoint;
            if (beta > 180) {
                //further_midPoint = pointGen(mountingPoint, slope_beta, further_dist, )[1];
                pointGen(mountingPoint, slope_beta, further_dist, pointCollector);
                further_midPoint = pointCollector[1];
            }
            else {
                pointGen(mountingPoint, slope_beta, further_dist, pointCollector);
                further_midPoint = pointCollector[0];
            }

            double further_edge = (heightPix[20] * std::tan(omega / 2)) / std::cos(ALPHA + (phi / 2));

            // Obtaining on ground triangle points
            pointGen(further_midPoint, -1 / slope_beta, further_edge, pointCollector);
            cv::Point point2 = pointCollector[1];
            cv::Point point3 = pointCollector[0];

            // plotting the points
            std::vector<cv::Point> pt = { mountingPoint, point2, point3 };
            genQuadImages(pt, cutQuadMask, cutQuadBrdr);
            // cv::cvtColor(cutQuadMask, cutQuadMask, cv::COLOR_BGR2GRAY);

            // code block to check if mounting point is directly viewing inside a bldg
            cv::Mat circleCheck(origImg.rows, origImg.cols, CV_8UC1, cv::Scalar(0));
            cv::circle(circleCheck, mountingPoint, 3, 255, 1);
            
            cv::bitwise_and(circleCheck, cutQuadMask, Check_step1);

            cv::bitwise_and(Check_step1, bldg_mask, Check_step2);

            std::vector<cv::Point> nonzeroX;
            // std::cout<<"\nHere -1\n";
            cv::findNonZero(Check_step2, nonzeroX);

            if (nonzeroX.size() > 0) {
                // cv::imshow("nonzero", Check_step2);
                // k = cv::waitKey(0);
                // cv::destroyAllWindows();
                exits++;
                continue;
            }
            // --------------------------------#

            // BldgInQuad = cv2.bitwise_and(imgCopy, cameraRoadCoverage)
            // bldg_brdr, bldg_mask = getBorderContour(BldgInQuad)
            // std::cout<<"\n\nHEREA\n\n";
            cv::Mat selected_bldg_brdrs_gray;
            cv::bitwise_and(bldg_brdr_gray, cutQuadMask, selected_bldg_brdrs_gray); // Gray
            // cv::cvtColor(bldg_brdr, selected_bldg_brdrs_gray, cv::COLOR_BGR2GRAY);
            // std::cout<<"\n\nHEREB\n\n";
            // 
            std::vector<std::vector<cv::Point>> selected_edge_list;
            cv::findContours(selected_bldg_brdrs_gray, selected_edge_list, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);



            getRoadCoverageMask(selected_edge_list, mountingPoint, yLim, xLim, roadCoveredMask);

            cv::bitwise_and(roadCopy, cutQuadMask, cameraRoadCoverage, roadCoveredMask);
            cv::Mat demo;
            cv::bitwise_or(imgCopy, cameraRoadCoverage, demo);
            // cv::imshow("Coverage", demo);
            // k = cv::waitKey(0);
            // cv::destroyAllWindows();
            calls++;

            // find the updated area of camera coverage
            std::vector<std::vector<cv::Point>> cameraRoadCoverageContour;
            cv::findContours(cameraRoadCoverage, cameraRoadCoverageContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            // std::cout<<"\n\nHEREC\n\n";
            double area_sum = 0;
            for (auto contour : cameraRoadCoverageContour) {
                // std::cout<<"\n\nHERED\n\n";
                double area = cv::contourArea(contour);
                // std::cout<<"\n\nHERE-E\n\n";
                area_sum += area;
            }

            if (area_sum > maxArea) {
                // std::cout<<"\nEntered Here\n";
                maxArea = area_sum;
                fp1 = mountingPoint;
                fp2 = point2;
                fp3 = point3;
                
                cv::bitwise_or(imgCopy, cameraRoadCoverage, maxCameraRoadCoverage);
                // std::cout<<"\nExited Here\n";
            }
        }
    }
    std::cout<<"\n\n Calls = ";
    std::cout<<calls<<std::endl;
    std::cout<<"\n\n unCalls = ";
    std::cout<<uncalls<<std::endl;
    std::cout<<"\n\n exits = ";
    std::cout<<exits<<std::endl;
    std::cout<<"\n";
    std::vector<cv::Point> finalPointList = {fp1, fp2, fp3};
    cv::polylines(maxCameraRoadCoverage, finalPointList, true, cv::Scalar(255));
    cv::imshow("Display window 2", maxCameraRoadCoverage);
    k = cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}

//NOTES: 
//Wrong final output
//Not exponentially faster
//Potential Improvements
//Increase angle step size
//sort in ascending order after first run ?