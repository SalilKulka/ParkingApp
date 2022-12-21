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
for x in range(30,60):
    heights.append(x*0.1)
# heights = [x*0.1 for x in range(30,60)] #average height of light poles is 9 to 14 feet ~ 4.2m max
centerPoints = []
scalePoints = []
scaleConst = 10
theta = (66.75*math.pi)/180 #converting to radians
mountingPoints = []
area_scaling_factor = 1.4
imagePath = 'C:\\Users\\Salil kulkarni\\Desktop\\TARQ\\Parking\\map3.PNG'
selectedPoints = []

def getMountPoints(img):
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

    # findcontours
    # contours, hier = cv2.findContours(combined_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    selectedPoints = []
    
    # f = open("C:\\Users\\Salil kulkarni\\Desktop\\TARQ\\Parking\\coord.txt","w")

    for x in range(len(combined_mask)):
        for y in range(len(combined_mask[x])):
            if(combined_mask[x][y]==255):
                # selectedPoints.append((x,y))
                # f.write(str(x) + "," + str(y) + "\n")
                mountingPoints.append((x,y))

def getBuildingBorders(img):
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
    
# nping it
    # mask = np.ones(combined_mask.shape, dtype=np.uint8) * 255
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # findcontours
    cnts=cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255,0,255), thickness=1)

    # selectedPoints = []
    
    # f = open("C:\\Users\\Salil kulkarni\\Desktop\\TARQ\\Parking\\coord.txt","w")
    mountingPoints.clear()
    for y in range(len(img)):
        for x in range(len(img[y])):
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            if(b == 255 and g == 0 and r == 255):
                mountingPoints.append((x,y))
                # f.write(str(x) + "," + str(y) + "\n")
    # print(mountingPoints)

def get_resized_for_display_img(img):
    screen_w, screen_h = GetSystemMetrics(0), GetSystemMetrics(1)
    #print("screen size",screen_w, screen_h)
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

def centre_gen(point1,slope,distance):
    slope_squared = pow(slope,2)
    x = point1[0] + distance*abs(pow(1/(1+slope_squared),0.5))
    y = point1[1] + distance*slope*abs(pow(1/(1+slope_squared),0.5))
    centerPoints.append((int(x),int(y)))
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
# displaying the coordinates
# on the Shell
        # print(x, ' ', y)

# displaying the coordinates
# on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +str(y), (x,y), font,1, (255, 0, 0), 2)
        cv2.imshow('image', img)

def click_event_1(event, x, y, flags, params):

# checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        midLine.append((x, y))
# displaying the coordinates
# on the Shell
        # print(x, ' ', y)

# displaying the coordinates
# on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +str(y), (x,y), font,1, (255, 0, 0), 2)
        cv2.imshow('image', img)

# checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
        scalePoints.append((x,y))

        

# displaying the coordinates
# on the Shell
        # print(x, ' ', y)

# displaying the coordinates
# on the image window
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
#     img = cv2.imread(imagePath, 1)
#     img = get_resized_for_display_img(img)
# # displaying the image
#     cv2.imshow('image', img)

# # setting mouse handler for the image
# # and calling the click_event_1() function
#     cv2.setMouseCallback('image', click_event_0)
#     # wait for a key to be pressed to exit
#     cv2.waitKey(0)
    getBuildingBorders(img)
    # Distance between selected midline extreme points
    distance = abs(pow(pow(midLine[0][0] - midLine[1][0],2) + pow(midLine[0][1] - midLine[1][1],2),0.5))

    diameter = (distance/numCircles) 
    slope = ( midLine[1][1]- midLine[0][1])/(midLine[1][0]-midLine[0][0])

    # Pixel distance of scale in image
    scale = abs(pow(pow(scalePoints[0][0] - scalePoints[1][0],2) + pow(scalePoints[0][1] - scalePoints[1][1],2),0.5))
    # actual distance = (scale constant)*(obtained magnitude)/scale
    for index in range(numCircles):
        if(index==0):
            image = cv2.circle(img,centre_gen(midLine[0],slope,diameter/2),int(diameter*area_scaling_factor/2),(0,255,0))
        else:
            image = cv2.circle(img,centre_gen(centerPoints[index-1],slope,diameter),int(diameter*area_scaling_factor/2),(0,255,0))

    image = get_resized_for_display_img(image)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    # print(f"Working Distance - {workingDistance}")
    f = open("C:\\Users\\Salil kulkarni\\Desktop\\TARQ\\Parking\\point_data.txt","w")
    f.write(f"radius - {0.5*scaleConst*diameter/scale} \n")
    for center in centerPoints:
        for point in mountingPoints:
            # print(f"Point - {point}")
            f.write(f"point - {point[0]}, {point[1]}; ")
            for height in heights:
                f.write(f"center - {center[0]}, {center[1]}; ")
                heightCoord = (height/scaleConst)*scale     # Height in pixels
                resultantDistance = abs(pow(pow(point[1]-center[1],2) + pow(point[0]-center[0],2) + pow(heightCoord,2),0.5))
                distance_to_center = abs(pow(pow(point[1]-center[1],2) + pow(point[0]-center[0],2),0.5))
                alpha = math.atan(distance_to_center/heightCoord)
                # print(f"Distance of point from center - {resultantDistance}")
                Radius_fov = (diameter*area_scaling_factor/2)/math.cos(alpha)    # 
                workingDistance = Radius_fov/math.tan(theta/2)
                f.write(f"height - {height}; ")
                f.write(f"Working Distance - {scaleConst*(workingDistance/scale)}; ")
                f.write(f"Resultant Distance - {scaleConst*(resultantDistance/scale)}")
                f.write("\n---x---\n")
                if(distanceCompare(resultantDistance,workingDistance)):
                    print(f"Point - {point} at height - {height} is at a distance of {scaleConst*(resultantDistance/scale)}")
                    print(f"Working Distance - {scaleConst*(workingDistance/scale)}")
                    selectedPoints.append((point[0],point[1],heightCoord))
    finalImage = cv2.imread(imagePath,1)
    finalImage = get_resized_for_display_img(finalImage)
    print(selectedPoints)
    for index in range(numCircles):
        finalImage = cv2.circle(finalImage,centerPoints[index],int(diameter*(area_scaling_factor/2)),(0,255,0))
    for index in range(len(selectedPoints)):
        finalImage = cv2.circle(finalImage,(selectedPoints[index][0],selectedPoints[index][1]),10,(30,255,120))
        # cv2.putText(finalImage, str(selectedPoints[index], selectedPoints[index][0], cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
    cv2.imshow('image', finalImage)
    cv2.waitKey(0)
# close the window
    # cv2.destroyAllWindows()
    # for point in mountingPoints: