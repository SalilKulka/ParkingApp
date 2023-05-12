import numpy as np
import cv2
from libc.math cimport sqrt, fabs, pow, M_PI, HUGE_VAL
import math
cimport cython


cdef class Point:

    cdef int x, y

    def __init__(self, x, y):
        self.x = x
        self.y = y


#   p1 is mounting point (1d) and should be converted np array
#   p2 is 2d array of integer points
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def selectFarthestPoint(int[:] p1,int[:,:] ptList):
    #returns either of p2 or p3 depending on which point is farthest away from p1 (mounting point)
    cdef float maxDistance = 0
    cdef int p[2]
    cdef int i
    cdef len_ptList = ptList.shape[1]
    for i in range(len(len_ptList)):
        d1 = fabs(sqrt(pow(p1[0]-ptList[i][0],2)+pow(p1[1]-ptList[i][1],2)))
        if(d1>maxDistance):
            p[0] = ptList[i][0]
            p[1] = ptList[i][1]
            maxDistance = d1
        retP = [p[0],p[1]]
    return retP


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def cornerInclude(int[:] p1, int[:] p2, int[:] mountingPoint, int[:,:] cutQuadBorderPtsOnimgOutline):
    cdef int h = 400
    cdef int w = 500
    cdef unsigned char[:,:] blank = np.zeros((400,500), dtype = "uint8")
    cdef int[:,:] final_list
    cdef int[:] y_vals
    cdef int[:] x_vals
    cdef int i
    if((rangeCheck(p1[1],h) and not rangeCheck(p2[1], h)) or (rangeCheck(p2[1],h) and not rangeCheck(p1[1], h))):   #if both of them do not lie on the same border
        bottomBorder = cv2.line(blank, (0, h), (w,h), 255, 5)
        intersection = cv2.bitwise_and(bottomBorder, cutQuadBorderPtsOnimgOutline)
        y_vals = np.nonzero(intersection)[0]
        x_vals = np.nonzero(intersection)[1]
        final_list = np.array([[x_vals[i], y_vals[i]] for i in range(len(x_vals))])
        return selectFarthestPoint(mountingPoint, final_list)


    elif((rangeCheck(p1[1],0) and not rangeCheck(p2[1], 0)) or (rangeCheck(p2[1],0) and not rangeCheck(p1[1], 0))):   #if both of them do not lie on the same border
        topBorder = cv2.line(blank, (0, 0), (w,0), 255, 5)
        intersection = cv2.bitwise_and(topBorder, cutQuadBorderPtsOnimgOutline)
        y_vals = np.nonzero(intersection)[0]
        x_vals = np.nonzero(intersection)[1]
        final_list = np.array([[x_vals[i], y_vals[i]] for i in range(len(x_vals))])
        return selectFarthestPoint(mountingPoint, final_list)


    elif((rangeCheck(p1[0],w) and not rangeCheck(p2[0], w)) or (rangeCheck(p2[0],w) and not rangeCheck(p1[0], w))):   #if both of them do not lie on the same border
        rightBorder = cv2.line(blank, (w, 0), (w,h), 255, 5)
        intersection = cv2.bitwise_and(rightBorder, cutQuadBorderPtsOnimgOutline)
        y_vals = np.nonzero(intersection)[0]
        x_vals = np.nonzero(intersection)[1]
        final_list = np.array([np.array([x_vals[i], y_vals[i]]) for i in range(len(x_vals))])
        return selectFarthestPoint(mountingPoint, final_list)

    elif((rangeCheck(p1[0],0) and not rangeCheck(p2[0], 0)) or (rangeCheck(p2[0],0) and not rangeCheck(p1[0], 0))):   #if both of them do not lie on the same border
        leftBorder = cv2.line(blank, (0, 0), (0,h), 255, 5)
        intersection = cv2.bitwise_and(leftBorder, cutQuadBorderPtsOnimgOutline)
        y_vals = np.nonzero(intersection)[0]
        x_vals = np.nonzero(intersection)[1]
        final_list = np.array([[x_vals[i], y_vals[i]] for i in range(len(x_vals))])
        return selectFarthestPoint(mountingPoint, final_list)

        
    return (-1,-1)

def cluster2Point_c(unsigned char[:,:] clusterImg):
# Make the corners into one point

    # innitates the the single point corner arrays
    cdef int P[30][2]
    cdef int cx,cy
    cdef int counter = 0

    # blank = np.zeros((400,500,3), dtype='uint8')
    white_mask = cv2.inRange(np.array(clusterImg), 255, 255)


    # combine masks
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(white_mask, cv2.MORPH_DILATE,kernel)


    cnts=cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for i in cnts:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = <int>(M['m10']/M['m00'])
            cy = <int>(M['m01']/M['m00'])
        # print(f"center - {cx},{cy}")
        P[counter] = [cx,cy]
        # cv2.circle(imageCopy, (cx, cy), 1, (255, 255, 255), -1)
        # cv2.drawContours(blank, [i], -1, 0, -1)
    Points = [P[i] for i in range(counter)]
    return Points


def rangeCheck(int checkAgainst, int val,float tolerance=0.01):
    if(val>=checkAgainst*(1-tolerance) and val<=checkAgainst*(1+tolerance)):
        return True
    return False


def pointGen(Point source, float m, float l):
    # m is the slope of line, and the
    # required Point lies distance l
    # away from the source Point
    cdef Point a = Point(0, 0)
    cdef Point b = Point(0, 0)
    cdef float dx,dy

    # slope is 0
    if m == 0:
        a.x = <int>(source.x + l)
        a.y = <int>(source.y)

        b.x = <int>(source.x - l)
        b.y = <int>(source.y)

    # if slope is infinite
    elif (m==HUGE_VAL):
        a.x = <int>(source.x)
        a.y = <int>(source.y + l)

        b.x = <int>(source.x)
        b.y = <int>(source.y - l)
    else:
        dx = (l / sqrt(1 + (m * m)))
        dy = m * dx
        a.x = <int>(source.x + dx)
        a.y = <int>(source.y + dy)
        b.x = <int>(source.x - dx)
        b.y = <int>(source.y - dy)
    
    return [[a.x,a.y],[b.x,b.y]]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def drawLine(float slope, int[:] point):    #img is a numpy array of uint8
    #draws line segment infinitely from edges of image
    cdef int y_lim = 400
    cdef int x_lim = 500

    cdef float px = -1
    cdef float py = -1
    cdef float qx = -1
    cdef float qy = -1

    cdef unsigned char[:,:] returnImage = np.zeros((400,500), dtype = 'uint8')


    if(slope==HUGE_VAL or -slope==HUGE_VAL):
        px = point[0]
        py = y_lim

        qx = point[0]
        qy = 0
    else:
        px = x_lim
        py = slope*x_lim + (point[1] - slope*point[0])

        qx = 0
        qy = (point[1] - slope*point[0])

    # cv2.line(returnImage, (int(px),int(py)),(int(qx), int(qy)), 255, 1)

    try:        
        cv2.line(returnImage, (<int>(px),<int>(py)),(<int>(qx), <int>int(qy)), 255, 1)
    except:
        # print(slope, px, py, qx, qy)
        pass
    return returnImage  


#
#   cutQuadBorder is 2d array of unsigned char 
#   mountingPoint is a 1d array of int (convert to np array before passing)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getRoadCoverageMask_c(selected_edge_list, unsigned char[:,:] cutQuadBorder, int[:] mountingPoint):
    cdef slope1, slope2 
    # cdef unsigned char[:,:] totalMask = np.zeros((400,500,3)[:2], dtype = 'uint8')
    cdef unsigned char[:,:] joiningLine_1,joiningLine_2,Point_1,Point_2
    cdef int x1,y1,x2,y2,a
    cdef int final_PointCoords_1[2]
    cdef int final_PointCoords_2[2]
    cdef int[:] y_vals
    cdef int[:] x_vals
    cdef int[:,:] pts
    totalMask = np.zeros((400,500), dtype = 'uint8')
    for selected_edge_contour in selected_edge_list:
        #mask related to a single contour, to which individual line blocking masks will be OR'ed to 
        contourMask = np.zeros((400,500), dtype = 'uint8')

        oneContourPic = np.zeros((400,500,3), dtype='uint8')
        cv2.drawContours(oneContourPic, [selected_edge_contour], -1, (255,255,255), thickness=1) #draws only selected contour for the loop
        oneContourPic = cv2.cvtColor(oneContourPic, cv2.COLOR_BGR2GRAY)   #convert drawn picture into grayscale

        oneContourPic = cv2.Canny(oneContourPic,50,150,apertureSize=3)

        # Apply HoughLinesP method to 
        # to directly obtain line end points
        lines = cv2.HoughLinesP(
                    oneContourPic, # Input edge image
                    1, # Distance resolution in pixels
                    M_PI/180, # Angle resolution in radians
                    threshold=11, # Min number of votes for valid line. Use lesser number of votes for smaller lines
                    minLineLength=5, # Min allowed length of line
                    maxLineGap=10 # Max allowed gap between line for joining them
                    )

        #same line is appearing in the pic twice, perhaps increase threshold (number of votes/points)
        if(lines is None):
            continue


        for points in lines:


            # Extracted points nested in the list
            x1 = points[0][0]
            y1 = points[0][1]
            x2 = points[0][2]
            y2 = points[0][3]


            # Step 4: For each line AND it with the interior corner points ??

            interiorPointCoords_1 = np.array([x1, y1])
            interiorPointCoords_2 = np.array([x2, y2])

            # Step 5: Isolate the points and their coordinates that intersect with the contour line
            # Step 6: Get the slope from the mounting point to those interior corners
            # Step 7: Drop a line from the interior corners to edge of image on a quad mask so you dont have to find intersection point with further edge
            
            slope1 = (mountingPoint[1] - interiorPointCoords_1[1])/(mountingPoint[0] - interiorPointCoords_1[0])
            slope2 = (mountingPoint[1] - interiorPointCoords_2[1])/(mountingPoint[0] - interiorPointCoords_2[0])

            joiningLine_1 = drawLine(slope1, interiorPointCoords_1)
            joiningLine_2 = drawLine(slope2, interiorPointCoords_2)

            
            
            Point_1 = cv2.bitwise_and(np.array(cutQuadBorder), np.array(joiningLine_1))

            PointCoords_1 = cluster2Point_c(Point_1)  #more than one cluster might form or no cluster might wrong
            # print(PointCoords_1)
            if(len(PointCoords_1)==0):  #weird case that you can't do anything about. Realistically this shouldn't exist, but it does
                continue
            if(len(PointCoords_1)==1):
                final_PointCoords_1 = PointCoords_1[0]
            elif(len(PointCoords_1)==2):
                final_PointCoords_1 = selectFarthestPoint(mountingPoint, PointCoords_1)
            elif(len(PointCoords_1)>2):
                y_vals = np.nonzero(Point_1)[0]
                x_vals = np.nonzero(Point_1)[1]
                pts = np.array([[x_vals[a], y_vals[a]] for a in range(len(x_vals))])
                final_PointCoords_1 = selectFarthestPoint(mountingPoint, pts)

            Point_2 = cv2.bitwise_and(np.array(cutQuadBorder), np.array(joiningLine_2))
            PointCoords_2 = cluster2Point_c(Point_2)  #more than one cluster might form or no cluster might wrong
            if(len(PointCoords_2)==0):  #weird case that you can't do anything about. Realistically this shouldn't exist, but it does
                continue
            if(len(PointCoords_2)==1):
                final_PointCoords_2 = PointCoords_2[0]
            elif(len(PointCoords_2)==2):
                final_PointCoords_2 = selectFarthestPoint(mountingPoint, PointCoords_2)
            elif(len(PointCoords_2)>2):
                y_vals = np.nonzero(Point_2)[0]
                x_vals = np.nonzero(Point_2)[1]
                pts = np.array([[x_vals[a], y_vals[a]] for a in range(len(x_vals))])
                final_PointCoords_2 = selectFarthestPoint(mountingPoint, pts)

            lineMask = np.zeros((400,500), dtype='uint8')
            pointSet = [interiorPointCoords_1, interiorPointCoords_2, final_PointCoords_2, final_PointCoords_1]
            # print(pointSet)
            inclusionCorner = cornerInclude(PointCoords_1, PointCoords_2)
            if(inclusionCorner[0]!=-1):
                pointSet.insert(3,inclusionCorner)


            # Step 8: Fill the quadrilateral made by the line contour, the lines from interior corners or the view quad side wall and the view quad further wall
            # There is a concern that the points are listed in not the correct order...
            blockPts = np.array( pointSet, dtype=np.int32) #original
            # blockPts = np.array( [interiorPointCoords_1, interiorPointCoords_2, PointCoords_2[0], PointCoords_1[0]], dtype=np.int32)  #working change by Atharva
            blockPts = blockPts.reshape((-1, 1, 2))
            lineMask = cv2.fillPoly(lineMask, pts=[blockPts],color=255)

            # Step 9: OR all the quadrilateral generated inside the loop on the view quad mask
            contourMask = cv2.bitwise_or(contourMask, lineMask)
        totalMask = cv2.bitwise_or(totalMask, contourMask)
    totalMask = cv2.bitwise_not(totalMask)
    return totalMask


