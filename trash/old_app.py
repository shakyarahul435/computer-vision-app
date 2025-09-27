import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import math 
import time

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
    '[T] Translation, Rotation, Scaling, [A] Calibrate Camera'
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

#AR


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
    
    elif mode == 'a':
        # camera calibration
        CHESSBOARD_SIZE = (9, 6)  # Number of internal corners (width, height)
        SQUARE_SIZE_MM = 25       # The real-world size of a square on your chessboard in mm
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)               # Create a grid of points in 3D space 9*6 = 54 points
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) 

        objp = objp * SQUARE_SIZE_MM

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        images_captured = 0
        TARGET_IMAGES = 20 # Number of images to capture for calibration

        last_capture_time = time.time()

        while images_captured < TARGET_IMAGES:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_corners, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)    
            # True if found, corners are the pixel coordinates of the corners Nx1x2

            display_frame = frame.copy()
            
            # If found, add object points, image points (after refining them)
            if ret_corners:
                # Draw the corners to give visual feedback
                cv.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)

                # Capture an image every 2 seconds to allow for repositioning
                if time.time() - last_capture_time > 2:
                    print(f"Found corners! Capturing image {images_captured + 1}/{TARGET_IMAGES}...")
                    
                    # Refine corner locations
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # Refine the corner locations to sub-pixel accuracy  
                    #-- lower projection error
                    # shape of corners2 is Nx1x2

                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    
                    images_captured += 1
                    last_capture_time = time.time()

                    frame_color_scale = cv.cvtColor(frame)

    # elif mode == 'r':
    #     h,w = frame.shape[:2]

    # elif mode == 'r':
    # # --- ArUco marker detection ---
    #     aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    #     parameters = cv.aruco.DetectorParameters()

    #     # Detect markers
    #     corners, ids, rejected = cv.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    #     if ids is not None:
    #         # Estimate pose for each detected marker
    #         rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
    #             corners, 0.05, mtx, dist
    #         )

    #         # Draw detected markers
    #         cv.aruco.drawDetectedMarkers(frame, corners, ids)

    #         # Draw axis for each marker
    #         for i in range(len(ids)):
    #             cv.drawFrameAxes(frame, cameraMatrix=mtx, distCoeffs=dist, rvecs[i], tvecs[i], 0.03)

    #     frame_color_scale = frame

    # elif mode == 'r':
    #     # Load calibration if not done yet
    #     global mtx, dist
    #     print('Displaying Trex Model')
    #     # trex = pywavefront.Wavefront('trex_model.obj', collect_faces=True)
    #     vertices = np.array([
    #         [0,0,0],
    #         [1,0,0],
    #         [1,1,0],
    #         [0,1,0],
    #         [0,0,-1],
    #         [1,0,-1],
    #         [1,1,-1],
    #         [0,1,-1]
    #     ], dtype=np.float32)

    #     # if mtx is None or dist is None:
    #     #     try:
    #     #         data = np.load("calibration_data.npz")
    #     #         mtx, dist = data["mtx"], data["dist"]
    #     #         print("Loaded calibration data from file.")
    #     #     except:
    #     #         print("Error: Calibration not done yet! Run mode 'a' first.")
    #     #         frame_color_scale = frame
    #     #         continue

    #     mtx = np.array([[800, 0, 320],
    #             [0, 800, 240],
    #             [0,   0,   1]], dtype=np.float32)
    #     dist = np.zeros((5,1))

    #     # --- ArUco marker detection ---
    #     aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    #     parameters = cv.aruco.DetectorParameters()

    #     corners, ids, rejected = cv.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    #     if ids is not None:
    #         # Estimate pose
    #         rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

    #         # cv.aruco.drawDetectedMarkers(frame, corners, ids)
    #         # for i in range(len(ids)):
    #         #     cv.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.03)

    #         axis_len = 0.05
    #         obj_pts = np.float32([
    #             [0,0,0], [axis_len,0,0], [axis_len,axis_len,0], [0,axis_len,0],
    #             [0,0,-axis_len], [axis_len,0,-axis_len], [axis_len,axis_len,-axis_len], [0,axis_len,-axis_len]
    #         ])
    #         img_pts, _ = cv.projectPoints(obj_pts, rvecs[0], tvecs[0], mtx, dist)
    #         img_pts = np.int32(img_pts).reshape(-1, 2)
    #         # Define cube faces by indices
    #         faces = [
    #             [0, 1, 2, 3],  # bottom
    #             [4, 5, 6, 7],  # top
    #             [0, 1, 5, 4],  # side 1
    #             [1, 2, 6, 5],  # side 2
    #             [2, 3, 7, 6],  # side 3
    #             [3, 0, 4, 7],  # side 4
    #         ]
    #         face_colors = [
    #             (255, 0, 0),    # Blue
    #             (0, 255, 0),    # Green
    #             (0, 0, 255),    # Red
    #             (255, 255, 0),  # Cyan
    #             (255, 0, 255),  # Magenta
    #             (0, 255, 255),  # Yellow
    #         ]
    #         # Draw filled faces
    #         for idx, face in enumerate(faces):
    #             cv.fillConvexPoly(frame, img_pts[face], face_colors[idx], lineType=cv.LINE_AA)
    #         # Draw black wireframe
    #         for face in faces:
    #             cv.polylines(frame, [img_pts[face]], isClosed=True, color=(0,0,0), thickness=2, lineType=cv.LINE_AA)

    #     frame_color_scale = frame



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

    # elif key == ord('a'):
    #     mode = 'a'



    elif key == ord('a'):
        # ----- Camera Calibration -----
        CHESSBOARD_SIZE = (9, 6)  # number of internal corners (width, height)
        SQUARE_SIZE_MM = 25       # real-world square size in mm
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare 3D object points
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE_MM

        objpoints = []  # 3D points in real world
        imgpoints = []  # 2D points in image plane

        TARGET_IMAGES = 20
        captured_images = 0
        last_capture_time = time.time()

        print("Starting callibration mode.. Show chessboard to the camera.")

        while captured_images < TARGET_IMAGES:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret_corners, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

            display_frame = frame.copy()

            if ret_corners:
                cv.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
                
                # Capture one image every 2 seconds to avoid duplicates
                if time.time() - last_capture_time > 2:
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    captured_images += 1
                    last_capture_time = time.time()
                    print(f"Captured image {captured_images}/{TARGET_IMAGES}")

            cv.imshow(frame_name, display_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Perform calibration once enough images are captured
        if captured_images == TARGET_IMAGES:
            ret_calib, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            print("Calibration done!")
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist)

        np.savez("calibration_data.npz", mtx=mtx, dist=dist)

        mode = 'n'
        print('Calibration finished.')

    # elif key == ord('r'):
    #     mode = 'r'

        


cap.release()
cv.destroyAllWindows()
