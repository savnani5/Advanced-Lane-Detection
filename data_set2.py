import cv2
import numpy as np
from matplotlib import pyplot as plt


def std_least_square(data_points):
    """
    Standard least squares implemetation:
    To fit a parabola we need to satisfy the equation y = ax^2 + bx + c
    """
    # For forming the X and Y array:
    X = []
    Y = []
    for point in data_points: # Note: DATA points are flipped
        X.append([point[1]**2, point[1], 1])
        Y.append(point[0])

    X = np.array(X)
    Y = np.array(Y)

    # Formula for fitting standard least squares (Xt*X)B = Xt*Y 
    xtx = np.dot(X.T,X)
    try:
        Xt_X_inv = np.linalg.pinv(np.dot(X.T,X))
        Xt_Y = np.dot(X.T,Y)
        result = np.dot(Xt_X_inv, Xt_Y)
        return result
    except:
        return []


def trapezium_coordinates(x_list, result_sls, y_limit):
    max_x, max_y = 0, 0
    max_x, min_y = 999999, 999999

    for x in x_list:
        # Only y plays the role, min/max x can switch!
        y = (result_sls[0])*x**2 + (result_sls[1])*x + result_sls[2]
        if y > max_y:
            if y > y_limit:
                max_y = y_limit
                max_x = x
            else:
                max_y = y
                max_x = x
        
        if y < min_y:
            if y < 0:
                min_y = 0
                min_x = x
            else:
                min_y = y
                min_x = x

    print((max_x, max_y), (min_x, min_y)) 
    # cv2.circle(line_img, (int(max_x), int(max_y)), 5, (0,255,0), 3)
    # cv2.circle(line_img, (int(min_x), int(min_y)), 5, (0,255,0), 3)

    return ((max_x, max_y), (min_x, min_y))

def turn_prediction(extreme_points, counter):
    bottom_centre_x, bottom_centre_y = (extreme_points[0][0] + extreme_points[2][0])//2, (extreme_points[0][1] + extreme_points[2][1])//2
    top_centre_x, top_centre_y = (extreme_points[1][0] + extreme_points[3][0])//2, (extreme_points[1][1] + extreme_points[3][1])//2
    # print(bottom_centre_x, bottom_centre_y)
    # print(top_centre_x, top_centre_y)
    slope = (top_centre_y - bottom_centre_y)/(top_centre_x - bottom_centre_x)
    # print(slope)
    if -8 < slope < 0:
        if counter < 400:
            return "Turn Left"
        else:
            return "Turn Right"
    elif 8 > slope > 0:
        if counter < 400:
            return "Turn Right"
        else:
            return "Turn Left"
    else:
        return "Move Straight"


def moving_avg(img, result_sls):
    # Moving weighted average filter on last 3 frame images
    if len(memory) <= 3:
        if result_sls != []:
            memory.append(img)
    else:
        if result_sls != []:
            memory.pop(0)
            memory.append(img)
    
    # avg = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    avg = memory[0]
    for i in range(1, len(memory)):
        avg = cv2.addWeighted(avg,0.8,memory[i],0.2,0)

    return avg


def pipeline(img, H, K, dist, counter):

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, K, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    # _____________________________________________________________________________

    # Warping the image
    img_dst = cv2.warpPerspective(dst, H, (dst.shape[1], dst.shape[0]))
    rows, cols, _ = img_dst.shape

    # Removing black vignette(due to undistortion) to narrow the ROI 
    img_dst[:, 0:int(cols*0.45)] = 0
    img_dst[:, int(cols*0.68):cols] = 0


    # Yellow mask (Separating the Yellow Colour from the image)
    hsv = cv2.cvtColor(img_dst, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([30,255,255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    yellow_line = cv2.bitwise_and(img_dst,img_dst, mask= mask)
    gray = cv2.cvtColor(yellow_line, cv2.COLOR_BGR2GRAY)
    ret, thresh_yellow = cv2.threshold(gray, 100,255,cv2.THRESH_BINARY)

    # ___________________________________________________

    gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    # Thresholding
    ret, thresh_white = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)

    # Adding the two images
    addition = cv2.bitwise_or(thresh_yellow, thresh_white)

    # Gaussian Blur
    blur = cv2.GaussianBlur(addition,(5,5),0)

    # Dummy image mask
    line_img = np.zeros((img_dst.shape[0], img_dst.shape[1], 3), dtype=np.uint8)

    # ____________________________________________________________________________________
    # Polynomial fitting

    # Getting the coordinates of white pixels
    w_indices = np.argwhere(thresh_white == 255)
    # print("w_ind", w_indices)

    result_sls = std_least_square(w_indices)
    # print(result_sls)
    if result_sls != []:
        wx = np.linspace(335, 385, 100)
        wy = (result_sls[0])*wx**2 + (result_sls[1])*wx + result_sls[2]


        draw_points = (np.asarray([wx, wy]).T).astype(np.int32)   # needs to be int32 and transposed
        # cv2.polylines(line_img, [draw_points], False, (0,0,255), 3)  # args: image, points, closed, color

        white_points = trapezium_coordinates(wx, result_sls, img_dst.shape[0])

        # X and Y range correction to avoid slope change(maxima/minima)
        wx = np.linspace(white_points[1][0], white_points[0][0], 100) # X will remain common for all methods
        wy = (result_sls[0])*wx**2 + (result_sls[1])*wx + result_sls[2]
        
        # To fill the gap in curve
        if abs(white_points[0][1] - img_dst.shape[0]) > 10:
            wx = np.append(wx, white_points[0][0])
            wy = np.append(wy, img_dst.shape[0])

    # _________________________________________________________________________________________
    # Getting the coordinates of yellow pixels
    y_indices = np.argwhere(thresh_yellow == 255)
    # print(y_indices)

    result_sls = std_least_square(y_indices)
    # print(result_sls)
    if result_sls != []:
        yx = np.linspace(290, 325, 100) # X will remain common for all methods
        yy = (result_sls[0])*yx**2 + (result_sls[1])*yx + result_sls[2]

        draw_points = (np.asarray([yx, yy]).T).astype(np.int32)   # needs to be int32 and transposed
        # cv2.polylines(line_img, [draw_points], False, (0,0,255), 3)  # args: image, points, closed, color


        yellow_points = trapezium_coordinates(yx, result_sls, img_dst.shape[0])

        # X and Y range correction to avoid slope change(maxima/minima)
        yx = np.linspace(yellow_points[1][0], yellow_points[0][0], 100) # X will remain common for all methods
        yy = (result_sls[0])*yx**2 + (result_sls[1])*yx + result_sls[2]

        # To fill the gap in curve
        if abs(yellow_points[0][1] - img_dst.shape[0]) > 10:
            yx = np.append(wx, yellow_points[0][0])
            yy = np.append(wy, img_dst.shape[0])

    if result_sls != []:
        # ____________________ Filling the Lane Mesh___________________________

        pts_left = np.array([np.transpose(np.vstack([yx, yy]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([wx, wy])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(line_img, np.int_(pts), (255,0,0))
        # __________________________________________________________________________
        
        i_line_img = line_img

        #_______________________Turn prediction__________________________________
        extreme_points = np.array([white_points[0], white_points[1], yellow_points[0], yellow_points[1]])
        print(extreme_points)
        
        turn = turn_prediction(extreme_points, counter)

        #_________Warping image back to original coordinates_______________________________
        line_img = cv2.warpPerspective(line_img, np.linalg.inv(H), (img_dst.shape[1], img_dst.shape[0]))
   
    else:
        turn = "Move straight"
        i_line_img = line_img
        extreme_points = []

    final = cv2.bitwise_or(dst, line_img)
    final = moving_avg(final, result_sls)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    final = cv2.filter2D(final, -1, kernel)

    cv2.putText(final, turn, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return final, img_dst, thresh_yellow, addition, thresh_white, i_line_img, line_img


if __name__ == "__main__":
    
    # Global variables
    # This will keep memory of images
    global memory
    memory = []

    #Camera Matrix
    K = np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
        [   0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
        [   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    #Distortion Coefficients
    dist = np.array([[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05, 2.20573263e-02]])

    # Calculating the homography Note: Calculated on undistorted image
    # source_points = np.float32([[317.2,203.6], [370.9,203.2], [470.4,291.6], [160.3, 291.6]])
    source_points = np.float32([[317.2,203.6], [370.9,203.2], [470.4,291.6], [160.3, 291.6]])
    # dst_points = np.float32([[250, 70], [325,70],  [325,291.6], [250, 291.6]])
    dst_points = np.float32([[300, 70], [370,70],  [370,291.6], [300, 291.6]])
    H = cv2.getPerspectiveTransform(source_points, dst_points)

    cap = cv2.VideoCapture('Data2/challenge_video.mp4')
    frame_width = int(cap.get(3)*0.5)
    frame_height = int(cap.get(4)*0.5)
    frametime = 100 # To control the video playback speed
    result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height)) 

    
    framerate = 20
    counter = 1
    
    try:
        while(cap.isOpened()):
            ret, frame = cap.read()
            width = int(frame.shape[1] * 0.5)
            height = int(frame.shape[0] * 0.5)
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
            
            try:
                # Calling the image processing pipeline
                final, img_dst, thresh_yellow, addition, thresh_white, i_line_img, line_img = pipeline(frame, H, K, dist, counter)
            except Exception as e:
                print("Error________________________________________________________",e)
                # break

            cv2.imshow('frame', final)
            result.write(final)
            if cv2.waitKey(framerate) & 0xFF == ord('q'):
                break
            counter +=1
    except: 
        pass
        
            
  
    cap.release()
    cv2.destroyAllWindows()