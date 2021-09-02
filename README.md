# Lane_Detection
In this project we aim to do simple Lane Detection to mimic Lane Departure
Warning systems used in Self Driving Cars. We are provided with two video
sequences, taken from a self-driving car. Our task will be to design an
algorithm to detect lanes on the road, as well as estimate the road curvature
to predict car turns. For detailed report on the project checkout [this link]().

## Input

![Input Data](https://drive.google.com/drive/folders/1r7ys7pS1fXXc7j13srBmU1pmoKM9wfr7?usp=sharing)

## Output

![Output Data](https://drive.google.com/drive/folders/1WzawWiSORhTyJsOCJmoqT7XxZU8T_e9s?usp=sharing)

## Pipeline

### Dataset1

![dataset1](git_gifs/img1.png)

![dataset1](git_gifs/img2.png)

### Dataset2

![dataset2](git_gifs/img3.png)

![dataset2](git_gifs/img4.png)


## How to Run the code
1) Change the directory for the first dataset in the **data_set1.py** file.
2) Run the below command to run the lane detection system on dataset1:
    ```python data_set1.py```
    
3) Change the directory for the second dataset in the **data_set2.py** file.\
4) Run the below command to run the lane detection system on dataset2:
    ```python data_set2.py```

## References
1) https://medium.com/@SunEdition/lane-detection-and-turn-prediction-algorithm-for-autonomous-vehicles-6423f77dc841
2) https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_table_of_contents_histograms/py_table_of_contents_histograms.html
3) https://www.geogebra.org/?lang=en
4) https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0
5) https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
6) https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
7) https://stackoverflow.com/questions/11270250/what-does-the-python-interface-to-opencv2-fillpoly-want-as-input
8) ENPM 673, Robotics Perception - Theory behind Homography Estimation

