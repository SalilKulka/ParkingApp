import cv2
import math
from win32api import GetSystemMetrics
import time
import numpy as np
#testing
imagePath = '.\\map3_1.PNG'

point = []
selectedPoints = []

def click_event_selectedpoints(event, x, y, flags, params):

# checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        selectedPoints.append([x,y])

# displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +str(y), (x,y), font,1, (255, 0, 0), 2)
        cv2.imshow('orig', image)

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
    cv2.imshow("image",img)
    cv2.waitKey(0)

    # write points in mounting list
    # # mountingPoints.clear()
    # for y in range(len(img)):
    #     for x in range(len(img[y])):
    #         b = img[y, x, 0]
    #         g = img[y, x, 1]
    #         r = img[y, x, 2]
      
       
if __name__=="__main__":

# reading the image
    img = cv2.imread(imagePath)
    img = get_resized_for_display_img(img)
    image = get_resized_for_display_img(img)
    # convert image to HSV
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    getBorderContour_text(img)

    low_yellow = (240,251,255)
    high_yellow = (240,251,255)

    low_gray = (244,243,241)
    high_gray = (244,243,241)

    cv2.imshow("orig",image)
    cv2.setMouseCallback('orig', click_event_0)
    cv2.waitKey(0)

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

    cv2.imshow("orig",image)
    cv2.setMouseCallback('orig', click_event_selectedpoints)
    cv2.waitKey(0)

    low_yellow = (240,251,255)
    high_yellow = (240,251,255)

    low_gray = (244,243,241)
    high_gray = (244,243,241)

    yellow_mask = cv2.inRange(img, low_yellow, high_yellow )
    gray_mask = cv2.inRange(img, low_gray, high_gray)

    # combine masks

    original_frame = cv2.imread(imagePath)
    original_frame = get_resized_for_display_img(original_frame)
    points = np.array(selectedPoints)
    blank = np.zeros(original_frame.shape[:2], dtype='uint8')
    poly_pts = np.array( points ,dtype=np.int32)
    poly_pts = poly_pts.reshape((-1, 1, 2))
    polymask = cv2.fillPoly(blank, pts=[poly_pts],color=255)
# pts - location of the corners of the roi
    masked = cv2.bitwise_and(original_frame,original_frame,mask=polymask)
    combined_mask = cv2.bitwise_or(yellow_mask, gray_mask)
    masked = cv2.bitwise_and(masked,masked,mask=combined_mask)

    # im = cv2.cvtColor (im, cv2.COLOR_BGR2GRAY)

    cv2.imshow("out", masked)
    cv2.waitKey(0)

    if(255 in masked):
        print(True)
    else:
        print(False)

    operatedImage = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # modify the data type
    # setting to 32-bit floating point
    operatedImage = np.float32(operatedImage)
    
    # apply the cv2.cornerHarris method
    # to detect the corners with appropriate
    # values as input parameters
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
    
    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)
    
    # Reverting back to the original image,
    # with optimal threshold value
    original_frame[dest > 0.01 * dest.max()]=[0, 0, 255]
    
    # the window showing output image with corners
    cv2.imshow('Image with Borders', original_frame)
    
    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


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


