import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import math 


def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'Clicked at: ({x},{y})')

def nothing(x):
    pass

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Camera cannot be opened.')
    exit()

mode = 'n' #default normal
frame_name = 'Camera Color Scale'
lines = [
    '[N] Normal, [G] GrayScale, [H] HSV,',
    '[C] Contrast/Brightness, [I] Histogram,',
    '[B] Gaussian Blur, [F] Bilateral Filter',
    '[E] Canny Edge Detection, [L] Hough Line Detection',
    '[Q] Quit Camera'
    ] 

cv.namedWindow(frame_name)
cv.setMouseCallback(frame_name, click_event)


# Brightness and Contrast Trackbar
BC_Tracker = False
trackBarBCCreated = False

# gaussian blur 
gaussianTracker = False
trackBarCreated = False

#bilateral 
bilateralTracker = False
bilateralTrackbarCreated = False

#translation
translationTracker = False
translationBarCreated = False


while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame Strem ending...")
        break

    if mode == 'g':
        # gray scale
        frame_color_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_color_scale = cv.cvtColor(frame_color_scale, cv.COLOR_GRAY2BGR)
    elif mode == 'h':
        # hsv scale
        frame_color_scale = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # frame_color_scale = cv.cvtColor(frame_color_scale, cv.COLOR_HSV2BGR)
    elif mode == 'c':
        # brightness and contrast
        if BC_Tracker:
            brightness = cv.getTrackbarPos('Brightness',frame_name) - 255
            contrast = cv.getTrackbarPos('Contrast',frame_name) / 10.0
            # frame_color_scale = cv.addWeighted(frame, contrast, brightness)
            frame_color_scale = cv.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            cv.imshow(frame_name, frame_color_scale)
        else:
            frame_color_scale = frame.copy()
    elif mode == 'i':
        # histogram
        frame_color_scale = frame.copy()
        hist_img = np.zeros((300, 512, 3), dtype=np.uint8)
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv.calcHist([frame], [i], None, [256], [0, 256])
            cv.normalize(hist, hist, 0, 300, cv.NORM_MINMAX)
            for j in range(1, 256):
                cv.line(hist_img,
                        (2*(j-1), 300 - int(hist[j-1])),
                        (2*j, 300 - int(hist[j])),
                        (255 if col=='b' else 0,
                         255 if col=='g' else 0,
                         255 if col=='r' else 0),
                        2)
        cv.imshow('Histogram', hist_img)
    elif mode == 'b':
        # gaussian blur
        if gaussianTracker:
            sigma = cv.getTrackbarPos('Sigma', frame_name)
            kernel = cv.getTrackbarPos('Kernel', frame_name)

            # kernel size to be odd number
            if kernel % 2 == 0:
                kernel += 1

            frame_color_scale = cv.GaussianBlur(frame, (kernel,kernel), sigma)
        else:
            frame_color_scale = frame.copy()
    
    elif mode == 'f':
        # bilateral Filter
        if bilateralTracker: 
            d = cv.getTrackbarPos('Diameter', frame_name)
            sigmaColor = cv.getTrackbarPos('SigmaColor', frame_name)
            sigmaSpace = cv.getTrackbarPos('SigmaSpace', frame_name)
            if d < 1:
                d = 1
            frame_color_scale = cv.bilateralFilter(frame, d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        else:
            frame_color_scale = frame.copy()

    elif mode == 'e':
        # Canny Edge Detection
        frame_color_scale = cv.Canny(frame, 100, 200)
    
    elif mode == 'l':
        # Hough Line Detection
        canny = cv.Canny(frame, 100, 200)
        edge = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        copy_edge = np.copy(edge)

        lines_Hough = cv.HoughLines(canny, 1, np.pi / 180, 150, None, 0,0 )

        if lines_Hough is not None:
            for i in range(0, len(lines_Hough)):
                rho = lines_Hough[i][0][0]
                theta = lines_Hough[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(copy_edge, pt1, pt2, (0,0,255), 3, cv.LINE_AA) 

        frame_color_scale = copy_edge


    elif mode == 't':
        # Image translation, rotation, and scaling
        if translationTracker:
            h, w = frame.shape[:2]

            # Read trackbar positions
            angle = cv.getTrackbarPos('Angle', frame_name) - 180  
            # range -180 to +180
            tx = cv.getTrackbarPos('Translate X', frame_name) - 150  
            # range -150 to +150
            ty = cv.getTrackbarPos('Translate Y', frame_name) - 100  
            # range -100 to +100
            scale = cv.getTrackbarPos('Scale', frame_name) / 100.0   
            # 0.2 – 2.0

            # Rotation + scaling around the center
            center = (w // 2, h // 2)
            M_rotate = cv.getRotationMatrix2D(center, angle, scale)
            rotated = cv.warpAffine(frame, M_rotate, (w, h))

            # Translation
            M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
            frame_color_scale = cv.warpAffine(rotated, M_translate, (w, h))
        else:
            frame_color_scale = frame.copy()

    else:
        frame_color_scale = frame

    # Labeling Text
    x, y0 = 20, 398
    line_height = 20 
    for i, line in enumerate(lines):
        y = y0 + i*line_height
        cv.putText(frame_color_scale, line, (x, y),
                cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0), 1, cv.LINE_AA)

    cv.imshow(frame_name, frame_color_scale)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        mode = 'g'
    elif key == ord('h'):
        mode = 'h'
    elif key == ord('n'):
        mode = 'n'
    elif key == ord('c'):
        mode = 'c'
        BC_Tracker = not BC_Tracker
        if BC_Tracker and not trackBarBCCreated:
            cv.createTrackbar('Brightness', frame_name, 50, 100, nothing)
            cv.createTrackbar('Contrast', frame_name, 50, 100, nothing)
            trackBarBCCreated = True

    elif key == ord('i'):
        mode = 'i'
    elif key == ord('b'):
        mode = 'b'
        gaussianTracker = not gaussianTracker
        if gaussianTracker and not trackBarCreated:  # create trackbars only once
            cv.createTrackbar('Sigma', frame_name, 1, 20, nothing)
            cv.createTrackbar('Kernel', frame_name, 3, 33, nothing)
            trackBarCreated = True

    elif key == ord('f'):
        # bilateral
        mode = 'f'
        bilateralTracker = not bilateralTracker
        if bilateralTracker and not bilateralTrackbarCreated:
            cv.createTrackbar('Diameter', frame_name, 5, 20, nothing)
            cv.createTrackbar('SigmaColor', frame_name, 75, 120, nothing)
            cv.createTrackbar('SigmaSpace', frame_name, 75, 120, nothing)
            bilateralTrackbarCreated = True

    elif key == ord('e'):
        mode = 'e'
    elif key == ord('l'):
        mode = 'l'
    elif key == ord('t'):
        mode = 't'
        translationTracker = not translationTracker
        if translationTracker and not translationBarCreated:
            angle_pos = cv.createTrackbar('Angle', frame_name, 180, 250, nothing)
            tx_pos = cv.createTrackbar('Translate X', frame_name, 5,150,nothing)
            ty_pos = cv.createTrackbar('Translate Y', frame_name, 5,150,nothing)
            scale_pos = cv.createTrackbar('Scale' ,frame_name, 25, 75, nothing)
            translationBarCreated = True

        

cap.release()
cv.destroyAllWindows()

