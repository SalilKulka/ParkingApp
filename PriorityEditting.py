# Importing all modules
import cv2
import math
import numpy as np

# Make a class to store x and y coordinates of points
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def toList(self):
        return [self.x, self.y]

# Make a class to store the view covered by a camera at each position  
class camView:
    def __init__(self, point0, point1, point2, cameraRoadCoverage):
        self.p0 = point0
        self.p1 = point1
        self.p2 = point2
        self.cameraRoadCoverage = cameraRoadCoverage

    def getNpPts(self):
        return np.array([self.p0.toList(), self.p1.toList(), self.p2.toList()], np.int32)

# Check if a point is between two others on the same line
def isBetweenCheck(a,c,b):
    softCheck = {math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)} + {math.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2)} - {math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)}
    return abs(softCheck) <= 0.5


def getMountingPoints(img):

    imageCopy = img.copy()

    # Upper and lower color limit customized for snazzy maps
    low_red = (55, 55, 255)

    # create masks
    red_mask = cv2.inRange(imageCopy, low_red, low_red)
    
    # combine masks
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE,kernel)

    # setting to 32-bit floating point
    operatedImage = np.float32(combined_mask)

    # apply the cv2.cornerHarris method to detect the corners with appropriate values as input parameters
    dest = cv2.cornerHarris(operatedImage, 2, 3, 0.04)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)

    # draw on the output image
    imageCopy[dest > 0.01 * dest.max()]=[255, 255, 255]

    return imageCopy

# Make the corners into one point
def cluster2Point(clusterImg):

    # Innitate the the single point corner arrays
    Points = []

    # Create masks for the corner clusters
    if(len(clusterImg.shape)==2):
        white_mask = cv2.inRange(clusterImg, 255, 255)
    elif(len(clusterImg.shape)==3):
        white_mask = cv2.inRange(clusterImg, (255,255,255), (255,255,255))

    # Combine masks
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(white_mask, cv2.MORPH_DILATE,kernel)

    # Approximate the clusters to a single point/pixel
    cnts=cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for i in cnts:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

        Points.append([cx,cy])

    # Return a list of points 
    return Points

def getRoadsnParkings(img):
  
    # define color ranges
    # blue_lower = (250,0,0)
    blue = np.array([255, 0, 0], dtype="uint8")

    # create mask
    blue_mask = cv2.inRange(img, blue, blue)

    return blue_mask

def pointGen(source, m, l):
    
    # m is the slope of line, and the required Point lies distance l away from the source Point
    a = Point(0, 0)
    b = Point(0, 0)

    # slope is 0
    if m == 0:
        a.x = source.x + l
        a.y = source.y

        b.x = source.x - l
        b.y = source.y

    # if slope is infinite
    elif math.isfinite(m) is False:
        a.x = source.x
        a.y = source.y + l

        b.x = source.x
        b.y = source.y - l
    else:
        dx = (l / math.sqrt(1 + (m * m)))
        dy = m * dx
        a.x = source.x + dx
        a.y = source.y + dy
        b.x = source.x - dx
        b.y = source.y - dy
    
    return [[a.x,a.y],[b.x,b.y]]

def detectCorner(image):

    # making a copy of the image to have the original image untouched in main loop
    imageSub = image.copy()

    # convert to gray and perform Harris corner detection
    gray = cv2.cvtColor(imageSub,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    #~~~~~~~~~~~~~#
    #for obtaining mounting points from red buildings img
    #~~~~~~~~~~~~~#
    dst = cv2.cornerHarris(gray,2,3,0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # threshold for an optimal value, it may vary depending on the image.
    imageSub[dst>0.01*dst.max()]=[0,0,255]

    return imageSub

# This function runs only once using new bgr values for the new image in low red.
def getBorderContour(img):

    # Upper and lower color limit customized for snazzy maps
    red = (55, 55, 255)

    # create masks
    red_mask = cv2.inRange(img, red, red)
    
    # combine masks
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE,kernel)

    blank = np.zeros(img.shape, dtype='uint8')

    # findcontours
    cnts=cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if(area>200):
            for eps in np.linspace(0.001, 0.01, 10):
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps * peri, True)
            
            # draw the approximated contour on the image  
            cv2.drawContours(blank, [approx], -1, (255,255,255), thickness=1)
  
    return blank, combined_mask

def drawLine(mountingPt, pointsCoords, y_lim, x_lim):

    #draws line segment infinitely from edges of image
    # x1,y1,x2,y2 = points2[0]

    p = [-1, -1]
    q = [-1, -1]

    if mountingPt[0]==pointsCoords[0]:

        p[0] = pointsCoords[0]
        p[1] = y_lim

        if not isBetweenCheck(mountingPt, [pointsCoords[0],pointsCoords[1]], p):
            p[0] = pointsCoords[0]
            p[1] = 0

    else:
        slope1 = (mountingPt[1] - pointsCoords[1])/(mountingPt[0] - pointsCoords[0])

        p[0] = x_lim
        p[1] = slope1*x_lim + (pointsCoords[1] - slope1*pointsCoords[0])

        if not isBetweenCheck(mountingPt, [pointsCoords[0],pointsCoords[1]], p):
            p[0] = 0
            p[1] = (pointsCoords[1] - slope1*pointsCoords[0])
            
    #__________________________________________________________________________________#

    if mountingPt[0]==pointsCoords[2]:

        q[0] = pointsCoords[2]
        q[1] = y_lim

        if not isBetweenCheck(mountingPt, [pointsCoords[2],pointsCoords[3]], q):
            q[0] = pointsCoords[2]
            q[1] = 0

    else:
        slope2 = (mountingPt[1] - pointsCoords[3])/(mountingPt[0] - pointsCoords[2])

        q[0] = x_lim
        q[1] = slope2*x_lim + (pointsCoords[3] - slope2*pointsCoords[2])

        if not isBetweenCheck(mountingPt, [pointsCoords[2],pointsCoords[3]], q):
            q[0] = 0
            q[1] = (pointsCoords[3] - slope2*pointsCoords[2])

    return p, q  

def getRoadCoverageMask(selected_edge_list, mountingPoint, yLim, xLim, img):

    # Initiate mask 
    totalMask = np.zeros(img.shape[:2], dtype = 'uint8')

    # Looping through all the edges of buildings inside camera coverage to perform manual ray traced image
    for selected_edge_contour in selected_edge_list:

        # Mask related to a single contour, to which individual line blocking masks will be OR'ed to 
        oneContourPic = np.zeros(img.shape[:2], dtype='uint8')

        # Draws only selected contour for the loop
        cv2.drawContours(oneContourPic, [selected_edge_contour], -1, 255, thickness=1) 
        oneContourPic = cv2.Canny(oneContourPic,50,150,apertureSize=3)
        

        # Apply HoughLinesP method to directly obtain line end points
        lines = cv2.HoughLinesP(
                    oneContourPic, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=11, # Min number of votes for valid line. Use lesser number of votes for smaller lines
                    minLineLength=5, # Min allowed length of line
                    maxLineGap=10 # Max allowed gap between line for joining them
                    )

        #same line is appearing in the pic twice, perhaps increase threshold (number of votes/points)
        if(lines is None):
            continue

        # Loop through the edges of buildings for fake ray traced image
        for points in lines:

            # Parse the points of the lines
            interiorPointCoords_1 = (points[0][0], points[0][1])
            interiorPointCoords_2 = (points[0][2], points[0][3])

            # Obtaining the point colinear with mounting point and building edge on the edge of the image
            brdrPoint_1, brdrPoint_2 = drawLine(mountingPoint, points[0], yLim, xLim)

            # Select the points of the poygon to exclude everything behind a building edge blocking the camera
            if (brdrPoint_1[0]==0 and brdrPoint_2[1]==0) or (brdrPoint_1[1]==0 and brdrPoint_2[0]==0):
                exPoints = [interiorPointCoords_1, brdrPoint_1, [0,0], brdrPoint_2, interiorPointCoords_2]
            elif (brdrPoint_1[0]==xLim and brdrPoint_2[1]==0) or (brdrPoint_1[1]==0 and brdrPoint_2[0]==xLim):
                exPoints = [interiorPointCoords_1, brdrPoint_1, [xLim,0], brdrPoint_2, interiorPointCoords_2]
            elif (brdrPoint_1[0]==0 and brdrPoint_2[1]==yLim) or (brdrPoint_1[1]==yLim and brdrPoint_2[0]==0):
                exPoints = [interiorPointCoords_1, brdrPoint_1, [0,yLim], brdrPoint_2, interiorPointCoords_2]
            elif (brdrPoint_1[0]==xLim and brdrPoint_2[1]==yLim) or (brdrPoint_1[1]==yLim and brdrPoint_2[0]==xLim):
                exPoints = [interiorPointCoords_1, brdrPoint_1, [xLim,yLim], brdrPoint_2, interiorPointCoords_2]
            else:
                exPoints = [interiorPointCoords_1, brdrPoint_1, brdrPoint_2, interiorPointCoords_2]
            
            
            # fill poly for excluded region
            exPoints = np.array(exPoints, dtype=np.int32)
            blockPts = exPoints.reshape((-1, 1, 2))
            totalMask = cv2.fillPoly(totalMask, pts=[blockPts],color=255)

    # Get the included region from the excluded region
    totalMask = cv2.bitwise_not(totalMask)

    return totalMask

def genQuadImages(pt):
    cutQuadMask3D = np.zeros(img.shape, dtype='uint8')
    cutQuadMask2D = np.zeros(img.shape[:2], dtype='uint8')
    cutQuadBorder = np.zeros(img.shape[:2], dtype='uint8')

    pt = pt.reshape((-1,1,2))
    
    cv2.fillPoly(cutQuadMask3D, [pt], (255,255,255))
    cv2.fillPoly(cutQuadMask2D, [pt], 255)

    # findcontours
    cnts=cv2.findContours(cutQuadMask2D, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        for eps in np.linspace(0.001, 0.01, 10):
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)
        
        # draw the approximated contour on the image  
        cv2.drawContours(cutQuadBorder, approx, -1, 255, thickness=1)
    return cutQuadMask3D, cutQuadBorder, cutQuadMask2D

def processMountingPoints(img, mountingPointsList, ALPHA, bldg_mask, bldg_brdr_gray):
    retList = []
    for mountingPoint in mountingPointsList:
        for beta in range (0,360,10):
            BETA = (beta*math.pi)/180
            if(beta==180 or beta==0 or beta==360):
                continue
            further_dist = heightPix[20]*math.tan(ALPHA + (phi/2))

            # slope of horizontal plane camera angle
            slope_beta = math.tan(BETA)

            # midpoints of closer and further edges
            if beta > 180:
                further_midPoint = pointGen(Point(mountingPoint[0],mountingPoint[1]), slope_beta, further_dist)[1]
            else:
                further_midPoint = pointGen(Point(mountingPoint[0],mountingPoint[1]), slope_beta, further_dist)[0]

            further_edge =(heightPix[20]*math.tan(omega/2))/math.cos(ALPHA+(phi/2))

            # Obtaining on ground triangle points
            point1 = [mountingPoint[0],mountingPoint[1]]
            point2 = pointGen(Point(further_midPoint[0],further_midPoint[1]), -1/slope_beta, further_edge)[1]
            point3 = pointGen(Point(further_midPoint[0],further_midPoint[1]), -1/slope_beta, further_edge)[0]
            
            # plotting the points
            pt = np.array([point1, point2, point3], np.int32)
            _, _, CameraCoverage2 = genQuadImages(pt)

            circleCheck = np.zeros(img.shape[:2], dtype = "uint8")
            cv2.circle(circleCheck, (mountingPoint[0], mountingPoint[1]), 3, 255, 1)
            Check_step1 = cv2.bitwise_and(circleCheck, CameraCoverage2)
            Check_step2 = cv2.bitwise_and(Check_step1, bldg_mask)
            nonzeroX, _ = np.nonzero(Check_step2)
            if len(nonzeroX)==0:
                pt = np.array([point1, point2, point3], np.int32)

                # get building borders inside viewing quadrilateral
                selected_bldg_brdrs_gray = cv2.bitwise_and(bldg_brdr_gray,CameraCoverage2) # Gray

                
                # 
                selected_edge_list = cv2.findContours(selected_bldg_brdrs_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                selected_edge_list = selected_edge_list[0] if len(selected_edge_list) == 2 else selected_edge_list[1]
                
                roadCoveredMask = getRoadCoverageMask(selected_edge_list, mountingPoint, yLim, xLim, img)

                retList.append(camView(Point(mountingPoint[0], mountingPoint[1]),Point(point2[0], point2[1]),Point(point3[0], point3[1]), roadCoveredMask))

    return retList
                
def main(img, camViewList, road):

    maxArea = 0
    imgCopy = img.copy()
    index = 0
    
    for view in camViewList:
            
        # Getting the camera coverage area
        cameraRoadCoverage, _, CameraCoverage2 = genQuadImages(view.getNpPts())
        cameraRoadCoverage = cv2.bitwise_and(road, CameraCoverage2, mask = view.cameraRoadCoverage)
        
        # find the updated are of camera coverage
        area_sum = np.count_nonzero(cameraRoadCoverage)

        if area_sum > maxArea:
            maxArea = area_sum
            fp2 = view.p1
            fp3 = view.p2
            mp = view.p0
            excludedRoad = cameraRoadCoverage
            maxCameraRoadCoverage = cv2.bitwise_or(cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY), cameraRoadCoverage)
            index = camViewList.index(view)

    
    return camView(mp, fp2, fp3,None), excludedRoad, maxCameraRoadCoverage, index

# Image path
imagePath = "Images\DubaiImg.png"
img = cv2.imread(imagePath)

# Dimensions of the image
yLim, xLim = img.shape[:2]

# Make a copy of the image to let user select priority areas
areaSelection = img.copy()

priorityAreas = []
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def selectArea(event,x,y,flags,param):
    global ix,iy,drawing,mode,preDrawState
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        preDrawState = areaSelection.copy()
    elif event == cv2.EVENT_MOUSEMOVE:
        preDrawState = areaSelection.copy()
        if drawing == True:
            if mode == True:
                cv2.rectangle(preDrawState,(ix,iy),(x,y),(0,255,0), 1)
            else:
                cv2.circle(preDrawState,(x,y),5,(0,0,255),-1)
        cv2.imshow("Priority Selection", preDrawState)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(areaSelection,(ix,iy),(x,y),(0,255,0),1)
        else:
            cv2.circle(areaSelection,(x,y),5,(0,0,255),-1)
        priorityAreas.append(((ix,iy),(x,y)))
        cv2.imshow("Priority Selection", areaSelection)

scalePoints = [(302, 373), (464, 373)] # points obtained from sample_scale.PNG map which is a google maps with the same dimensions
mountingPoints = []

theta = (66.75*math.pi)/180    #diagonal angle FOV of camera (GIVEN!!)
phi = 2*math.atan(0.8*math.tan(theta/2))  #angle of view larger side of camera resolution (4 in 4:3)
omega = 2*math.atan(0.6*math.tan(theta/2))     #angle of view larger side of camera resolution (3 in 4:3)
alpha = (75*math.pi)/180   #set later on in the code based on the height of the camera [angle of camera from negative z axis]

# Scale for pix to meter conversion
scaleConst = 20 

# Pixel distance of scale in image
# actual distance(m) = (scale constant)*(obtained magnitude)/scale
scale = abs(math.sqrt(pow(scalePoints[0][0] - scalePoints[1][0],2) + pow(scalePoints[0][1] - scalePoints[1][1],2)))

# heights = [x*0.1 for x in range(30,60)] #average height of light poles is 9 to 14 feet ~ 4.2m max
heights = [h*0.1 for h in range(30, 60)]    #3m to 6m
# Converting height in meter array to height in pixel array
heightPix = [h*scale/scaleConst for h in heights]

# reading the resized the image
img = cv2.imread(imagePath)

# obtaining the mask of the roads and parkings in the map image
road = getRoadsnParkings(img)   #image is grayscale

cv2.namedWindow('Priority Selection')
cv2.setMouseCallback('Priority Selection',selectArea)

cv2.imshow("Priority Selection", areaSelection)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(priorityAreas)

priorityMask = np.zeros(img.shape[:2],dtype = 'uint8')
nonPriorityMask = np.zeros(img.shape[:2],dtype = 'uint8')
for rect in priorityAreas:
    cv2.rectangle(priorityMask,rect[0],rect[1],255, -1)
cv2.subtract(road, priorityMask, nonPriorityMask)
cv2.bitwise_and(road, priorityMask, priorityMask)
cv2.imshow("Non Priority Mask", nonPriorityMask)
cv2.imshow("Priority Mask", priorityMask)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('win', road)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(imagePath)
ALPHA = 40*math.pi/180

mountingCluster = getMountingPoints(img)
mountingPointsList = cluster2Point(mountingCluster)

# getting building borders by using a mask function
bldg_brdr, bldg_mask = getBorderContour(img)
bldg_brdr_gray = cv2.cvtColor(bldg_brdr, cv2.COLOR_BGR2GRAY)
camViewList = processMountingPoints(img, mountingPointsList, ALPHA, bldg_mask, bldg_brdr_gray)

# angle of camera from negative of vertical axisS
img = cv2.imread(imagePath)

yLim, xLim = img.shape[:2]
roadLive = road.copy()

selectedViews = []

maxCameraRoadCoverage2 = img.copy()

maxCameraRoadCoverage = []

numCam = 3
camPlaced = 0

while(len(priorityAreas)):
    selectedView, excludedRoad1, maxCameraRoadCoverage_0, delIndex = main(img, camViewList, priorityMask)

    selectedViews.append(selectedView)
    del camViewList[delIndex]

    priorityMask = cv2.bitwise_and(priorityMask, cv2.bitwise_not(excludedRoad1))
    roadLive = cv2.bitwise_and(roadLive, cv2.bitwise_not(excludedRoad1))
    camPlaced = camPlaced + 1

    coverage = np.count_nonzero(priorityMask)

    if(camPlaced>=numCam or coverage<10):   #if number of cameras placed is equal to max number of cameras, or all priority 
        #priority areas have been covered ---> break out of the loop
        break

while(camPlaced<numCam):

    selectedView, excludedRoad1, maxCameraRoadCoverage_0, delIndex = main(img, camViewList, roadLive)

    selectedViews.append(selectedView)
    del camViewList[delIndex]

    roadLive = cv2.bitwise_and(roadLive, cv2.bitwise_not(excludedRoad1))
    camPlaced = camPlaced+1

# showing image
for x in range(len(selectedViews)):
     mp = selectedViews[x].p0.toList()
     fp2 = selectedViews[x].p1.toList()
     fp3 = selectedViews[x].p2.toList()
     cv2.circle(maxCameraRoadCoverage2, mp, 5, (255,0,0),2)
     pts = np.array([mp, fp2, fp3],np.int32)
     pts = pts.reshape((-1,1,2))
     cv2.polylines(maxCameraRoadCoverage2, [pts], True, (255,255,255))
cv2.imshow('included v2 static', maxCameraRoadCoverage2)
# cv2.imshow('excluded', GlobalCheck1)
cv2.waitKey(0)
cv2.destroyAllWindows()
