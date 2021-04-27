#Note: Put an edge case to remove the line on the right hand side (can be segragated by slope values)

import cv2
import numpy as np

def pipeline():
    pass

def line_selection():
    pass

def remove_jitters():
    pass

img = cv2.imread("Data1/data/0000000000.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = gray.shape
print(rows, cols)

# Narrowing the ROI below horizon
gray[0:int(rows*0.5), :] = 0

# Aggresive thresholding
ret,thresh = cv2.threshold(gray,210,255,cv2.THRESH_BINARY)

# Gaussian Blur
blur = cv2.GaussianBlur(thresh,(5,5),0)

# Morphological operation (erosion) to remove noise
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(blur,kernel,iterations = 1)
# opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)


edges = cv2.Canny(erosion,130,200)
# edges = cv2.Canny(opening,150,200)

# Dilation to make edges thicker
# dilation = cv2.dilate(edges,kernel,iterations = 1)



# Hough lines
line_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

lines = cv2.HoughLinesP(edges, rho=4, theta=np.pi/180, threshold=30, minLineLength=100, maxLineGap=180)
# for line in lines:
#     print("line", line)
#     for x1,y1,x2,y2 in line:
#         slope = (y2-y1)/(x2-x1)
#         print("slope", slope)
#         if slope < 4 and slope > -4:
#             cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)
#         else:
#             pass

try:
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    thresh_slope = 4   # To remove outliers, The straight pole has a very big slopee (almost a vertical line)
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            print(slope)
            if slope < thresh_slope and slope > 0.4:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope > -thresh_slope and slope < -0.4:
                l_slope.append(slope)
                l_lane.append(line)
        
        y_global_min = min(y1,y2,y_global_min)


    if((len(l_lane) == 0) or (len(r_lane) == 0)):
            print ('no lane detected')
            # return 1

    try:
        l_slope_mean = np.mean(l_slope,axis =0)
        r_slope_mean = np.mean(r_slope,axis =0)
        l_mean = np.mean(np.array(l_lane),axis=0)
        r_mean = np.mean(np.array(r_lane),axis=0)
    except:
        print("Division by zero")

    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    print("lb", l_b, "r_b", r_b)

    l_x1 = int((y_global_min - l_b)/l_slope_mean) 
    l_x2 = int((y_max - l_b)/l_slope_mean)   
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_b)
        r_y1 = int((r_slope_mean * r_x1 ) + r_b)
        l_y2 = int((l_slope_mean * l_x2 ) + l_b)
        r_y2 = int((r_slope_mean * r_x2 ) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    cv2.line(img,(l_x1,l_y1),(l_x2,l_y2),(0,0,255),6)
    cv2.line(img,(r_x1,r_y1),(r_x2,r_y2),(0,0,255),6)

except Exception as e:
    print(e)



cv2.imshow("image", img)
cv2.imshow("gray", gray)
cv2.imshow("thresh", thresh)
cv2.imshow("blur", blur)
cv2.imshow("erosion", erosion)
cv2.imshow("edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

if __name__ == "__main__":
    pass    