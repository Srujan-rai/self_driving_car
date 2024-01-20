import cv2
import numpy as np  
import matplotlib.pyplot as  plt

def make_coordinates(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])



def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        paramaters=np.polyfit((x1,x2),(y1,y2),1)
        slope=paramaters[0]
        intercept=paramaters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope , intercept))
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_coordinates(image,left_fit_average)
    right_line=make_coordinates(image,right_fit_average)
    return  np.array([left_line,right_line])


def canny(image):
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
 
    triangle = np.array([[
    (200, height),
    (550, 250),
    (1100, height),]], np.int32)
 
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            return line_image
image = cv2.imread('test_image.jpg')
frame=np.copy(image)   
    
cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    canny_image=canny(frame)
    cropped_image=region_of_interest(canny_image)
    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180, 100,np.array([]),minLineLength=40,maxLineGap=5)
    average_line=average_slope_intercept(frame,lines)
    line_image=display_lines(frame,average_line)
    combo_image=cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow("output",combo_image)
    if cv2.waitKey(1)& 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()