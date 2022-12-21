# importing the module
import cv2
import math
from win32api import GetSystemMetrics
import time
import numpy as np

# function to display the coordinates of
# of the points clicked on the image
midLine = []
numCircles = 10
heights = []
heightPix = []
for x in range(30,60):
    heights.append(x*0.1)
# heights = [x*0.1 for x in range(30,60)] #average height of light poles is 9 to 14 feet ~ 4.2m max
centerPoints = []
scalePoints = []
scaleConst = 10
mountingPoints = []
area_scaling_factor = 1.4
imagePath = 'C:\\Users\\Salil kulkarni\\Desktop\\TARQ\\Parking\\map3.PNG'
selectedPoints = []
buffer_length = 5   #buffer length before the midpoint from which viewing must begin
theta = (66.75*math.pi)/180    #diagonal angle FOV of camera (GIVEN!!)
phi = 2*math.atan(0.8*math.tan(theta/2))  #angle of view larger side of camera resolution (4 in 4:3)
omega = 2*math.atan(0.6*math.tan(theta/2))     #angle of view larger side of camera resolution (3 in 4:3)
alpha = 0   #set later on in the code based on the height of the camera [angle of camera from negative z axis]

point1 = [0,0]  #bottom right point of rectangle
point2 = [0,0]  #bottom left point of rectangle
point3 = [0,0]  #top left point of rectangle
point4 = [0,0]  #top right point of rectangle

def getBorderContour_road(img):
    low_yellow = (175,230,250)
    high_yellow = (185,235,255)

    low_gray = (228,225,223)
    high_gray = (234,230,229)

    # create masks
    yellow_mask = cv2.inRange(img, low_yellow, high_yellow )
    gray_mask = cv2.inRange(img, low_gray, high_gray)

    # combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, gray_mask)
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE,kernel)

    for x in range(len(combined_mask)):
        for y in range(len(combined_mask[x])):
            if(combined_mask[x][y]==255):
                # selectedPoints.append((x,y))
                # f.write(str(x) + "," + str(y) + "\n")
                mountingPoints.append((x,y))

def getBorderContour_text(img):

    # Upper and lower color limit
    low_yellow = (240,251,255)
    high_yellow = (240,251,255)

    low_gray = (244,243,241)
    high_gray = (244,243,241)

    # create masks
    yellow_mask = cv2.inRange(img, low_yellow, high_yellow )
    gray_mask = cv2.inRange(img, low_gray, high_gray)

    # combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, gray_mask)
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE,kernel)

    # findcontours
    cnts=cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255,0,255), thickness=1)

    # write points in mounting list
    mountingPoints.clear()
    for y in range(len(img)):
        for x in range(len(img[y])):
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            if(b == 255 and g == 0 and r == 255):
                mountingPoints.append((x,y))
                # f.write(str(x) + "," + str(y) + "\n")

def get_resized_for_display_img(img):

    # get screen height and width
    screen_w, screen_h = GetSystemMetrics(0), GetSystemMetrics(1)

    h,w,channel_nbr = img.shape
    # img get w of screen and adapt h
    h = h * (screen_w / w)
    w = screen_w
    if h > screen_h: #if img h still too big
        # img get h of screen and adapt w
        w = w * (screen_h / h)
        h = screen_h
    w, h = w*0.9, h*0.9 # because you don't want it to be that big, right ?
    w, h = int(w), int(h) # you need int for the cv2.resize
    return cv2.resize(img, (w, h))

def point_gen(point1,slope,distance,dxn=1):
    slope_squared = pow(slope,2)
    x = point1[0] + dxn*distance*abs(pow(1/(1+slope_squared),0.5))
    y = point1[1] + dxn*distance*slope*abs(pow(1/(1+slope_squared),0.5))
    # print(x,y)
    return (int(x),int(y))

def distanceCompare(result, camera_value):
    if(result >= camera_value) and (result <= camera_value*1.3):
        return True
    return False

def click_event_0(event, x, y, flags, params):

# checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        mountingPoints.append((x,y))

# displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +str(y), (x,y), font,1, (255, 0, 0), 2)
        cv2.imshow('image', img)

def click_event_1(event, x, y, flags, params):

# checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        midLine.append((x, y))

# displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +str(y), (x,y), font,1, (255, 0, 0), 2)
        cv2.imshow('image', img)

# checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
        scalePoints.append((x,y))

# displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +str(g) + ',' + str(r),(x,y), font, 1,(255, 255, 0), 2)
        cv2.imshow('image', img)

# driver function
if __name__=="__main__":

# reading the image
    img = cv2.imread(imagePath, 1)
    img = get_resized_for_display_img(img)
# displaying the image
    cv2.imshow('image', img)

# setting mouse handler for the image
# and calling the click_event_1() function
    scaleConst = int(input('Scale: '))
    cv2.setMouseCallback('image', click_event_1)
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    time.sleep(2)

    getBorderContour_text(img)
    # Distance between selected midline extreme points
    distance = abs(pow(pow(midLine[0][0] - midLine[1][0],2) + pow(midLine[0][1] - midLine[1][1],2),0.5))

    diameter = (distance/numCircles) 
    slope_midline = ( midLine[1][1]- midLine[0][1])/(midLine[1][0]-midLine[0][0])

    # Pixel distance of scale in image
    scale = abs(pow(pow(scalePoints[0][0] - scalePoints[1][0],2) + pow(scalePoints[0][1] - scalePoints[1][1],2),0.5))
    # actual distance = (scale constant)*(obtained magnitude)/scale

    image = get_resized_for_display_img(image)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    for height in heights:
        heightPix.append(height*scale/scaleConst)
    
    #the slope associated w/ line of sight of camera

    #distance b/w midpoints of closest edge and farthest edge of projected rectangle
    full_mag_rect_dist = h*(math.tan(alpha+(phi/2)) - math.tan(alpha - (phi/2)))

    #finding mid point of closest edge of first projected rectangle 
    mid_point_closer_edge = point_gen(midLine[0], slope_midline, scale*buffer_length/scaleConst,-1)
    #finding mid point of farther edge of first projected rectangle 
    mid_point_further_edge = point_gen(mid_point_closer_edge, slope_midline, full_mag_rect_dist)

    #half of the closer edge length = (h*tan(w/2))/cos(alpha-(phi/2))
    half_mag_closer_edge = (h*math.tan(omega/2))/math.cos(alpha-(phi/2))
    #half of the further edge length = (h*tan(w/2))/cos(alpha+(phi/2))
    half_mag_further_edge =(h*math.tan(omega/2))/math.cos(alpha+(phi/2))

    slope_camera_view = (mountingPoint[1]- mid_point_closer_edge[1])/(mountingPoint[0]-mid_point_closer_edge[0])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #BIG DISCLAIMER - will have to change 1/slope
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
    point1 = point_gen(mid_point_closer_edge, -1/slope_camera_view, half_mag_closer_edge)
    point2 = point_gen(mid_point_closer_edge, -1/slope_camera_view, half_mag_closer_edge,-1)

    point3 = point_gen(mid_point_further_edge, -1/slope_camera_view, half_mag_further_edge)
    point4 = point_gen(mid_point_further_edge, -1/slope_camera_view, half_mag_further_edge,-1)

    pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,255))

    cv2.imshow('image', finalImage)
    cv2.waitKey(0)
# close the window
    # cv2.destroyAllWindows()
    # for point in mountingPoints: