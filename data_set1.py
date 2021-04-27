import cv2
import glob
import numpy as np

def moving_avg(curr_lane):

    # Moving average filter on last 5 frames
    if len(memory) <= 5:
        if curr_lane != []:
            memory.append(curr_lane)
    else:
        if curr_lane != []:
            memory.pop(0)
            memory.append(curr_lane)

    # print("memory", memory)
    left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2 = 0,0,0,0,0,0,0,0
    for lane in memory:
        left_x1 += lane[0][0]
        left_y1 += lane[0][1]
        left_x2 += lane[1][0]
        left_y2 += lane[1][1]
        right_x1 += lane[2][0]
        right_y1 += lane[2][1]
        right_x2 += lane[3][0]
        right_y2 += lane[3][1]

    left_x1 = (left_x1//len(memory))
    left_y1 = left_y1//len(memory)
    left_x2 = left_x2//len(memory)
    left_y2 = left_y2//len(memory)
    right_x1 = right_x1//len(memory)
    right_y1 = right_y1//len(memory)
    right_x2 = right_x2//len(memory)
    right_y2 = right_y2//len(memory)

    return [(left_x1,left_y1), (left_x2,left_y2), (right_x1,right_y1),(right_x2,right_y2)]


def turn_prediction(extreme_points):

    bottom_centre_x, bottom_centre_y = (extreme_points[1][0] + extreme_points[3][0])//2, (extreme_points[1][1] + extreme_points[3][1])//2
    top_centre_x, top_centre_y = (extreme_points[0][0] + extreme_points[2][0])//2, (extreme_points[0][1] + extreme_points[2][1])//2
    # print(bottom_centre_x, bottom_centre_y)
    # print(top_centre_x, top_centre_y)
    slope = (top_centre_y - bottom_centre_y)/(top_centre_x - bottom_centre_x)
    # print(slope)

    if -5 < slope < 0:
        return "Turn Left"
    elif 5 > slope > 0:
        return "Turn Right"
    else:
        return "Move Straight"


def line_selection(img, lines):
    """ This function is to find the prominent lane lines from all the lines detected in the video sequence
        Note: Put an edge case to remove the line on the right hand side (can be segragated by slope values)
    """
    try:
        # To remove outliers, The straight pole on right hand sidehas a very big slopee (almost a vertical line)
        upper_thresh_slope_r = 3
        upper_thresh_slope_l = 1.5   
        lower_thresh_slope = 0.4
        y_min, y_max = img.shape[0], img.shape[0] 
        left_slope, left_lane, right_slope, right_lane = [],[],[],[]
        
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1)
                # print(slope)  

                if slope < upper_thresh_slope_r and slope > lower_thresh_slope:
                    right_slope.append(slope)
                    right_lane.append(line)
                elif slope > -upper_thresh_slope_l and slope < -lower_thresh_slope:
                    left_slope.append(slope)
                    left_lane.append(line)
            
            y_min = min(y_min, y1, y2)
        
        left_slope_mean = np.mean(left_slope,axis=0)
        right_slope_mean = np.mean(right_slope,axis=0)
        left_mean = np.mean(np.array(left_lane),axis=0)
        right_mean = np.mean(np.array(right_lane),axis=0)
 
        # b = y - m*x
        left_b = left_mean[0][1] - (left_slope_mean * left_mean[0][0])
        right_b = right_mean[0][1] - (right_slope_mean * right_mean[0][0])

        left_x1 = int((y_min - left_b)/left_slope_mean) 
        left_x2 = int((y_max - left_b)/left_slope_mean)   
        right_x1 = int((y_min - right_b)/right_slope_mean)
        right_x2 = int((y_max - right_b)/right_slope_mean)

        if left_x1 > right_x1:
            left_x1 = int((left_x1+right_x1)/2)
            right_x1 = left_x1
            left_y1 = int((left_slope_mean * left_x1 ) + left_b)
            right_y1 = int((right_slope_mean * right_x1 ) + right_b)
            left_y2 = int((left_slope_mean * left_x2 ) + left_b)
            right_y2 = int((right_slope_mean * right_x2 ) + right_b)
        else:
            left_y1 = y_min
            left_y2 = y_max
            right_y1 = y_min
            right_y2 = y_max
        
        return [(left_x1,left_y1), (left_x2,left_y2), (right_x1,right_y1),(right_x2,right_y2)]
    
    except Exception as e:
        # print(e)
        return []

def pipeline(img, K, dist):
    """ Lane detection Pipeline 
    """
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, K, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    # print(rows, cols)

    # Narrowing the ROI below horizon
    gray[0:int(rows*0.5), :] = 0
    gray[:, 0:int(cols*0.25)] = 0

    # Aggresive thresholding
    ret,thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(thresh,(5,5),0)

    # Morphological operation (erosion) to remove noise
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(blur,kernel,iterations = 1)

    # Canny Edge Detection
    edges = cv2.Canny(erosion,130,200)

    # Hough lines
    line_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    # lines = cv2.HoughLinesP(edges, rho=4, theta=np.pi/180, threshold=30, minLineLength=50, maxLineGap=180)
    lines = cv2.HoughLinesP(edges, rho=4, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=180)
    # for line in lines:
    #     print("line", line)
    #     for x1,y1,x2,y2 in line:
    #         slope = (y2-y1)/(x2-x1)
    #         print("slope", slope)
    #         if (slope < 4 and slope > 0.4) or (slope > -4 and slope < -0.4):
    #             # print("slope", slope)
    #             cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),4)
    #         else:
    #             pass

    lanes = line_selection(dst,lines)
    try:
        lanes = moving_avg(lanes)
    except:
        pass

    if len(lanes) !=0:    
        # cv2.line(dst,(lanes[0][0], lanes[0][1]),(lanes[1][0], lanes[1][1]),(0,0,255),6)
        # cv2.line(dst,(lanes[2][0], lanes[2][1]),(lanes[3][0], lanes[3][1]),(0,0,255),6)
        
        bottom_centre_x, bottom_centre_y = (lanes[1][0] + lanes[3][0])//2, (lanes[1][1] + lanes[3][1])//2
        top_centre_x, top_centre_y = (lanes[0][0] + lanes[2][0])//2, (lanes[0][1] + lanes[2][1])//2
        turn = turn_prediction(lanes)
        cv2.line(dst,(bottom_centre_x, bottom_centre_y),(top_centre_x, top_centre_y),(0,0,255),3)

        # Filling the lane mesh
        new_img = np.zeros((dst.shape[0], dst.shape[1], 3), dtype=np.uint8)
        cnt = np.array([[[lanes[0][0], lanes[0][1]],[lanes[1][0], lanes[1][1]],[lanes[3][0], lanes[3][1]], [lanes[2][0], lanes[2][1]]]], dtype=np.int32 )
        cv2.fillPoly(new_img, cnt, (255,0,0))
        dst = cv2.bitwise_or(dst, new_img) 

    return turn, dst, line_img, gray, thresh, blur, erosion, edges


if __name__ == "__main__":

    # This will keep memory of lanes
    global memory
    memory = []
    framerate = 20

    #Camera Matrix
    K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02], 
        [0.000000e+00, 9.019653e+02, 2.242509e+02], 
        [0.000000e+00, 0.000000e+00, 1.000000e+00]])

    #distortion coefficients
    dist = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])



    for img in sorted(glob.glob('Data1/data/*.png')):
    
        img = cv2.imread(img)
        # print(img.shape)
    
        turn, dst, line_img, gray, thresh, blur, erosion, edges = pipeline(img, K, dist)
        cv2.putText(dst, turn, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("lane_detection", dst)
        # cv2.imshow("gray", gray)
        # cv2.imshow("thresh", thresh)
        # cv2.imshow("blur", blur)
        # cv2.imshow("erosion", erosion)
        # cv2.imshow("edges", edges)



        if cv2.waitKey(framerate) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


