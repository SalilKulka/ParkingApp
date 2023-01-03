
import cv2
import math
from win32api import GetSystemMetrics
import time
import numpy as np
import os


# Image path
imagePath = 'Images'
imageName = "sample3.PNG"
imagePath = os.path.join(imagePath,imageName)
mountingPoints = []
quadPoints = []


# Steps to do test/implement ray masking
# 1. Plot out viewing quadrilateral
# 2. Plot out mounting point
# 3. Mask out for building contours
# 4. Find corners of building contours
# 
# Taking the nearest corner, find the slope of the line joining the mounting point and this nearest corner. Find the intersection point of this line and the farthest edge of the quadrilateral (What if farthest edge is out of bounds ??)

def getBorderContour_text(img):
#using new bgr values for the new image in low red.

    img_c = img.copy()
    # Upper and lower color limit customized for snazzy maps
    low_red = (55, 55, 255)
    high_yellow = (242,251,256)

    low_gray = (241,241,241)
    high_gray = (244,243,241)

    # create masks
    red_mask = cv2.inRange(img, low_red, low_red )
    
    # combine masks
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE,kernel)

    blank = np.zeros(img.shape, dtype='uint8')

    masked = cv2.bitwise_and(img,img,mask=combined_mask)

    # findcontours
    cnts=cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        # cv2.drawContours(img, [c], -1, (255,0,255), thickness=1)
        area = cv2.contourArea(c)
        if(area>200):
            for eps in np.linspace(0.001, 0.01, 10):
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps * peri, True)
            
            # draw the approximated contour on the image  
            cv2.drawContours(blank, [approx], -1, (255,255,255), thickness=1)
            # cv2.drawContours(blank, [c], -1, 255, thickness=1)


    # cv2.imshow("image",img)
    # cv2.waitKey(0)

    return blank,masked

def click_MountingnScale_points(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        mountingPoints.append((x,y))

        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +str(y), (x,y), font,1, (255, 0, 0), 2)
        cv2.imshow('img', img)


    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
        quadPoints.append((x,y))

        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +str(y),(x,y), font, 1,(0, 0, 255), 2)
        cv2.imshow('img', img)

def getCorners(masked, blocksize = 3):

    operatedImage = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # modify the data type
    # setting to 32-bit floating point
    # operatedImage = cv2.fastNlMeansDenoisingColored(operatedImage,None,10,10,7,21) #uncomment if you feel image is noisy (not needed)
    operatedImage = np.float32(operatedImage)

    dest = cv2.cornerHarris(operatedImage, blocksize, 5, 0.07)  #increase the second parameter ~ blocksize to get more of the corner shape out

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)

    size = masked.shape

    suro = np.zeros(size, dtype='uint8')
    suro[dest > 0.01 * dest.max()]=[255, 255, 255]

    blank = np.zeros(masked.shape[:2], dtype='uint8')
    poly_pts = np.array( quadPoints ,dtype=np.int32)
    poly_pts = poly_pts.reshape((-1, 1, 2))
    polymask = cv2.fillPoly(blank, pts=[poly_pts],color=255)

    suro = cv2.bitwise_and(suro,suro,mask=polymask)

    return suro

def findClusterCenters(img, excludedCornerPoints = []):

    corner_centers = []

    blank = np.zeros(img.shape, dtype='uint8')
    # blank_copy = blank.copy()

    # create masks
    white_mask = cv2.inRange(img, (255,255,255), (255,255,255))

    # combine masks
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(white_mask, cv2.MORPH_DILATE,kernel)


    cnts=cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for i in cnts:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        # print(f"center - {cx},{cy}")
        if([cx,cy] not in excludedCornerPoints):
            cv2.circle(blank, (cx, cy), 1, (255, 255, 255), -1)
            corner_centers.append([cx,cy])

    return corner_centers,blank

def getCorners_w_exclude(img, cornerPoints=[]):
    
    operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)

    # for pts in poly_pts:
    #     dest[pts[0]][pts[1]] = 0
    # Reverting back to the original image,
    # with optimal threshold value
    size = img.shape

    suro = np.zeros(size, dtype='uint8')
    suro[dest > 0.01 * dest.max()]=[255, 255, 255]

    points,suro = findClusterCenters(suro,cornerPoints)

    return suro, points




img = cv2.imread(imagePath)
cv2.imshow("img",img)
cv2.setMouseCallback('img', click_MountingnScale_points)

cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.imread(imagePath)
bldg_brdrs, bldg_mask, cntr_list = getBorderContour_text(img)


blank = np.zeros(img.shape[:2], dtype='uint8')
poly_pts = np.array(quadPoints,dtype=np.int32)
poly_pts = poly_pts.reshape((-1, 1, 2))
polymask = cv2.fillPoly(blank, pts=[poly_pts],color=255)

highlighted_corners = getCorners(bldg_brdrs)
cornerPoints,point_corners = findClusterCenters(highlighted_corners)

highlighted_corners = cv2.bitwise_and(highlighted_corners, highlighted_corners, mask = polymask)


result = cv2.bitwise_and(bldg_brdrs,highlighted_corners, )

result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


cnts=cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]


slope = (c[4,0,1] - c[2,0,1])/(c[4,0,0] - c[2,0,0]) 
print(slope)
slope = (c[4,0,1] - c[0,0,1])/(c[4,0,0] - c[0,0,0]) 
print(slope)


#Doing further calculation based on cnts[0], but in main code loop over all contours

#eqs of side edges of quadrilateral
slope_left_edge = (quadPoints[2][1] - quadPoints[0][1])/(quadPoints[2][0] - quadPoints[0][0])
slope_right_edge = (quadPoints[3][1] - quadPoints[1][1])/(quadPoints[3][0] - quadPoints[1][0])
c_left_edge = quadPoints[2][1] - slope_left_edge*quadPoints[2][0]
c_right_edge = quadPoints[1][1] - slope_left_edge*quadPoints[1][0]

#eqs of 2 lines from corner point contour
slope_corner_edge_1 = (c[4,0,1] - c[2,0,1])/(c[4,0,0] - c[2,0,0]) 
slope_corner_edge_2 = (c[4,0,1] - c[0,0,1])/(c[4,0,0] - c[0,0,0]) 
c_corner_edge_1 = 
c_corner_edge_2 = 



mountingPoint = mountingPoints[0]

slope_farthest_edge = (quadPoints[3][1]-quadPoints[2][1])/(quadPoints[3][0]-quadPoints[2][0]) #slope of farthest edge line

c2 = quadPoints[3][1] - slope_farthest_edge*quadPoints[3][0] #constant of farthest edge line

for cornerPoint in cornerPoints:
    #if corner point is already blacked out, do not do operations w/ it
    if(img[cornerPoint[1], cornerPoint[2], 0] == 0 and img[cornerPoint[1], cornerPoint[2], 1] == 0 and img[cornerPoint[1], cornerPoint[2], 2] == 0):
        continue
    slope = (mountingPoint[1] - cornerPoint[1])/(mountingPoint[0] - cornerPoint[0])


    c1 = mountingPoint[1] - slope*mountingPoint[0] #constant of line joining mounting point and corner = y1 - mx1

    #have obtained one point for the blacked out region
    x_intercept = (c2-c1)/(slope - slope_farthest_edge)
    y_intercept = slope*x_intercept + c1
#  gay