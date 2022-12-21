import cv2
import math
from win32api import GetSystemMetrics
import time
import numpy as np

imagePath = 'C:\\Users\\Salil kulkarni\\Desktop\\TARQ\\Parking\\map3.PNG'

point = []

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
    # convert image to HSV
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



    # define color ranges
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

    selectedPoints = []
    
    # f = open("C:\\Users\\Salil kulkarni\\Desktop\\TARQ\\Parking\\coord.txt","w")

    for x in range(len(combined_mask)):
        for y in range(len(combined_mask[x])):
            if(combined_mask[x][y]==255):
                selectedPoints.append((x,y))
                # f.write(str(x) + "," + str(y) + "\n")

    # f.close()
    



    # displaying the image
    # result = cv2.bitwise_and(img,img, mask= mask)
    cv2.imshow('Mask',img)

    # setting mouse handler for the image
    # and calling the click_event_1() function
    # cv2.setMouseCallback('Mask', click_event_0)
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
