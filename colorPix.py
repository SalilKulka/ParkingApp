import cv2

# Load the image
img = cv2.imread("Images\sample3.png")

# Get the shape of the image
height, width, _ = img.shape

# Define the starting point for the rays
start_x, start_y = 50, 50

# Create a blank image for the rays
ray_img = img.copy()

# Define the color threshold for red
red_threshold = (200, 0, 0)

# Iterate over all pixels in the image
for x in range(width):
    for y in range(height):
        # Check if the current pixel is red
        if img[y, x][2] > red_threshold[2]:
            # Draw a line from the starting point to the red pixel
            cv2.line(ray_img, (start_x, start_y), (x, y), (0, 0, 255), 1)
            # Break the loop when the first red edge is detected
            break

# Show the image with the rays
cv2.imshow("Rays", ray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
