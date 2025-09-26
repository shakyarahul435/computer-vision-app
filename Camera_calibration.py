import numpy as np 
import cv2 as cv
import time


def calibrate_camera(cap, frame_name):
    CHESSBOARD_SIZE = (9, 6)
    SQUARE_SIZE_MM = 25
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints, imgpoints = [], []
    captured_images = 0
    TARGET_IMAGES = 20
    last_capture_time = time.time()

    print("Starting calibration mode.. Show chessboard to the camera.")
    while captured_images < TARGET_IMAGES:
        ret, frame = cap.read()
        if not ret: break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret_corners, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        display_frame = frame.copy()

        if ret_corners:
            cv.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
            if time.time() - last_capture_time > 2:
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                captured_images += 1
                last_capture_time = time.time()
                print(f"Captured image {captured_images}/{TARGET_IMAGES}")

        cv.imshow(frame_name, display_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    if captured_images == TARGET_IMAGES:
        ret_calib, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Calibration done!")
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
        np.savez("calibration_data.npz", mtx=mtx, dist=dist)
    print("Calibration finished.")
