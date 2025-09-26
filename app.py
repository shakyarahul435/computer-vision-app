import cv2 as cv
import numpy as np
import math
from AR import AugmentedRealityHandler
from Panorama import PanoramaHandler
from Camera_calibration import calibrate_camera

# ----------------------- Utility Functions -----------------------
def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'Clicked at: ({x},{y})')

def nothing(x):
    pass

def create_trackbar(frame_name, name, default, max_val):
    cv.createTrackbar(name, frame_name, default, max_val, nothing)

def reset_window(frame_name):
    """Destroy and recreate window to clear trackbars."""
    cv.destroyWindow(frame_name)
    cv.namedWindow(frame_name)
    cv.setMouseCallback(frame_name, click_event)

# ----------------------- Filters / Transformations -----------------------
def apply_grayscale(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

def apply_hsv(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2HSV)

def adjust_brightness_contrast(frame, frame_name):
    brightness = cv.getTrackbarPos('Brightness', frame_name) - 255
    contrast = cv.getTrackbarPos('Contrast', frame_name) / 10.0
    return cv.convertScaleAbs(frame, alpha=contrast, beta=brightness)

def apply_gaussian_blur(frame, frame_name):
    sigma = cv.getTrackbarPos('Sigma', frame_name)
    kernel = cv.getTrackbarPos('Kernel', frame_name)
    if kernel % 2 == 0: kernel += 1
    return cv.GaussianBlur(frame, (kernel, kernel), sigma)

def apply_bilateral_filter(frame, frame_name):
    d = max(1, cv.getTrackbarPos('Diameter', frame_name))
    sigmaColor = cv.getTrackbarPos('SigmaColor', frame_name)
    sigmaSpace = cv.getTrackbarPos('SigmaSpace', frame_name)
    return cv.bilateralFilter(frame, d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

def apply_canny(frame):
    return cv.Canny(frame, 100, 200)

def apply_hough_lines(frame):
    canny = cv.Canny(frame, 100, 200)
    edge = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
    lines_Hough = cv.HoughLines(canny, 1, np.pi / 180, 150)
    if lines_Hough is not None:
        for rho_theta in lines_Hough:
            rho, theta = rho_theta[0]
            a, b = math.cos(theta), math.sin(theta)
            x0, y0 = a*rho, b*rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(edge, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    return edge

def apply_translation_rotation_scaling(frame, frame_name):
    h, w = frame.shape[:2]
    angle = cv.getTrackbarPos('Angle', frame_name) - 180
    tx = cv.getTrackbarPos('Translate X', frame_name) - 150
    ty = cv.getTrackbarPos('Translate Y', frame_name) - 100
    scale = cv.getTrackbarPos('Scale', frame_name) / 100.0

    M_rotate = cv.getRotationMatrix2D((w//2, h//2), angle, scale)
    rotated = cv.warpAffine(frame, M_rotate, (w, h))
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv.warpAffine(rotated, M_translate, (w, h))

def apply_histogram(frame):
    """Return original frame + color histogram side by side."""
    channels = cv.split(frame)  # B, G, R channels
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red

    # Create histogram canvas
    hist_img = np.zeros((435, 555, 3), dtype=np.uint8)
    bin_width = int(round(555 / 256))

    # Loop over each channel
    for chan, col in zip(channels, colors):
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        cv.normalize(hist, hist, 0, 400, cv.NORM_MINMAX)

        for i in range(1, 256):
            cv.line(hist_img,
                    (bin_width*(i-1), 400 - int(hist[i-1])),
                    (bin_width*i, 400 - int(hist[i])),
                    col, 2)

    # Resize frame to match histogram height
    frame_resized = cv.resize(frame, (555, 435))

    # Stack frame + histogram horizontally
    combined = np.hstack((frame_resized, hist_img))
    return combined

# ----------------------- Main Loop -----------------------

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Camera cannot be opened.')
        return

    frame_name = 'Camera Color Scale'
    cv.namedWindow(frame_name)
    cv.setMouseCallback(frame_name, click_event)

    mode = 'n'

    ar_handler = AugmentedRealityHandler(
            calibration_file="calibration_data.npz",
            model_path="trex_model.obj",   
            marker_length=0.12,            # trex size in meters
            model_scale_factor=0.0004,
            rotate_model=True
        )

    panorama_handler = PanoramaHandler()

    lines = [
        '[N] Normal, [G] GrayScale, [H] HSV,',
        '[C] Contrast/Brightness, [I] Histogram,',
        '[B] Gaussian Blur, [F] Bilateral Filter',
        '[E] Canny Edge Detection, [L] Hough Line Detection',
        '[T] Translation, Rotation, Scaling, [P] for Panorama',
        '[A] Calibrate Camera, [R] Augumented Reality, ',
        '[Q] Quit Camera'
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Stream ending...")
            break

        # ----------------------- Mode Handling -----------------------
        if mode == 'g':
            frame_color_scale = apply_grayscale(frame)
        elif mode == 'h':
            frame_color_scale = apply_hsv(frame)
        elif mode == 'c':
            frame_color_scale = adjust_brightness_contrast(frame, frame_name)
        elif mode == 'i':
            frame_color_scale = apply_histogram(frame)
        elif mode == 'b':
            frame_color_scale = apply_gaussian_blur(frame, frame_name)
        elif mode == 'f':
            frame_color_scale = apply_bilateral_filter(frame, frame_name)
        elif mode == 'e':
            frame_color_scale = apply_canny(frame)
        elif mode == 'l':
            frame_color_scale = apply_hough_lines(frame)
        elif mode == 't':
            frame_color_scale = apply_translation_rotation_scaling(frame, frame_name)
        elif mode == 'p':
            frame_color_scale = panorama_handler.process_frame(frame)
        elif mode == 'r':
            frame_color_scale = frame.copy()
            frame_color_scale = ar_handler.draw_3d_model(frame_color_scale, frame_name)
        else:
            frame_color_scale = frame.copy()

        # ----------------------- Overlay Text -----------------------
        for i, line in enumerate(lines):
            cv.putText(frame_color_scale, line, (33, 316+i*20),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)

        cv.imshow(frame_name, frame_color_scale)

        key = cv.waitKey(1) & 0xFF

        if panorama_handler.is_active() and panorama_handler.handle_key(key):
            continue

        if key == ord('q'):
            break
        elif key in [ord('n'), ord('g'), ord('h'), ord('c'), ord('i'),
                     ord('b'), ord('f'), ord('e'), ord('l'), ord('t'), ord('a'), ord('r'), ord('p')]:
            mode = chr(key)

            reset_window(frame_name)  # clears old trackbars

            # Recreate trackbars for active mode only
            if mode == 'c':
                create_trackbar(frame_name, 'Brightness', 50, 510)
                create_trackbar(frame_name, 'Contrast', 10, 50)
            elif mode == 'b':
                create_trackbar(frame_name, 'Sigma', 1, 20)
                create_trackbar(frame_name, 'Kernel', 3, 33)
            elif mode == 'f':
                create_trackbar(frame_name, 'Diameter', 5, 20)
                create_trackbar(frame_name, 'SigmaColor', 75, 120)
                create_trackbar(frame_name, 'SigmaSpace', 75, 120)
            elif mode == 't':
                create_trackbar(frame_name, 'Angle', 180, 360)
                create_trackbar(frame_name, 'Translate X', 150, 300)
                create_trackbar(frame_name, 'Translate Y', 100, 200)
                create_trackbar(frame_name, 'Scale', 100, 200)
            elif mode == 'a':
                calibrate_camera(cap, frame_name)
                ar_handler._load_calibration()
                mode = 'n'
            elif mode == 'p':
                panorama_handler.set_active(True)

    cap.release()
    cv.destroyAllWindows()


# ----------------------- Run -----------------------
if __name__ == "__main__":
    main()




