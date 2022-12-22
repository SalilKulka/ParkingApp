import cv2
import math
from win32api import GetSystemMetrics
import time
import numpy as np
#testing
imagePath = 'C:\\Users\\Salil kulkarni\\Desktop\\TARQ\\Parking\\map3.PNG'

point = []

def isContourBad(c):
    peri = cv2.arcLength(c, True)
    if(peri<7):
        return False
    return True

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

def click_event_0(event, x, y, flags, params):

# checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(b) + ',' +str(g) + ',' + str(r),(x,y), font, 1,(255, 255, 0), 2)
        cv2.imshow('image', img)

        
       
if __name__=="__main__":

# reading the image
    img = cv2.imread(imagePath)
    img = get_resized_for_display_img(img)
    image = get_resized_for_display_img(img)
    # convert image to HSV
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



    # define color ranges
    low_yellow = (255,255,255)
    high_yellow = (255,255,255)

    # create masks
    yellow_mask = cv2.inRange(img, low_yellow, high_yellow )

    # combine masks
    kernel = np.ones((3,3), dtype=np.uint8)
    combined_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_DILATE,kernel)



    # # Using canny
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(gray, 50, 80)

    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(image, contours, -1, 255, -1)

    # cv2.imshow('canny', image)

    # cv2.waitKey(0)


    # findcontours
    cnts=cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        # if(isContourBad(c)):
        cv2.drawContours(img, [c], -1, (255,0,255), thickness=1)

    mask = np.ones(img.shape[:2], dtype="uint8") * 255

    for contour in cnts:
        area = cv2.contourArea(contour)
        if area > 4000:
            cv2.drawContours(mask, [contour], -1, 0, -1)

    mask = cv2.bitwise_not(mask)

    # image = cv2.bitwise_subtract(img, img, mask=mask)

    
    # f.close()
    cv2.imshow('Mask',mask)
    cv2.waitKey(0)

    # for x in range(len(image)):
    #     for y in range(len(image[x])):
    #         if(image[x][y][0] == 255 and image[x][y][1] == 255 and image[x][y][2] == 255 ):
    #             img[x,y] = (255,0,255)
    #             # f.write(str(x) + "," + str(y) + "\n")

    # cv2.imshow('image',img)
    # cv2.waitKey(0)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Find the contours in the image
    # _, contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Set a threshold for the minimum area of the encircled regions
    # threshold_area = 20

    # # Iterate through the contours and remove any that have an area less than the threshold
    # for c in contours:
    #     if cv2.contourArea(c) < threshold_area:
    #         cv2.drawContours(image, [c], -1, (0,0,0), -1)


