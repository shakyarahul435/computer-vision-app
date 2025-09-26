# Computer Vision: README File for Camera Application 📷 #

**1. Create a virtual environment:**
- python -m venv env   (Virtual environment is created naming env)
- env\Scripts\Activate  (Activate the virtual environment env)

**2. Install required libraries in environment:**
- opencv-python
- numpy
- matplotlib
- open
- opencv-contrib-python   &nbsp;&nbsp;&nbsp; (For aruco libraries)



**3. Run python file: python app.py**
- CV2 frame will pop-up
- Firstly, Opens in Normal Frame or Press **[N]** for Normal Frame (RGB)
- Press **[G]** for gray scale frame,
- Press **[H]** for HSV scale frame,
- Press **[C]** for Contrast/Brightness
- Press **[I]** for Histogram
- Press **[B]** for Gaussian Blur
- Press **[F]** for Bilateral Filter 
- Press **[E]** for Canny Edge Detection
- Press **[L]** for Line Detection using Hough Transform
- Press **[P]** for Panaroma
    - Press **[Spacebar]** to click the image (max upto 5 image)
    - Press **[S]** to stich the image
    - Press **[R]** to reset the clicked image
- Press **[T]** for Translation, Rotation and Scaling
- Press **[A]** for Calibrate Camera
- Press **[R]** for Augmented Reality  (Trex Display)
- Finally, Press **[Q]** for Quiting the frame

**# Notes:**
- Make sure your camera is working.
- Panorama stiching works best on overlapping images.
- Augmented Reality (AR) requires running Camera Calibration first before displaying 3D Model.
- Camera Calibration data is stored in [calibration_data.npz](calibration_data.npz) 
- Trex 3D Model file is stored in [trex_model.obj](trex_model.obj)

**For More Information:**

<img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail" width="20" height="20"> Contact Me Via:  st125982@ait.asia


🌱**Published By:**
<br/> **---Rahul Shakya---**
