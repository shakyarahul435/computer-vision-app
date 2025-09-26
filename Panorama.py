import cv2 as cv
import numpy as np
import time

class PanoramaHandler:
    """Handles panorama creation and image stitching with manual processing."""
    
    def __init__(self):
        """Initialize panorama handler."""
        self.active = False
        self.capturing = False
        self.frames = []
        self.panorama = None
        self.max_frames = 5
        self.reproj_threshold = 4.0
        self.current_frame_count = 0
        self.preview_window_name = "Captured Frame Preview"
        self.current_frame = None  # Store current frame for manual capture
        self.panorama_window_name = "Panorama"
        
        # Feature detector and matcher
        self.detector = cv.SIFT_create()
        self.matcher = cv.BFMatcher()
    
    def set_active(self, active):
        """
        Set the active state of panorama mode.
        
        Args:
            active (bool): Whether panorama mode is active
        """
        self.active = active
        if not active:
            self.reset()
    
    def is_active(self):
        """
        Check if panorama mode is active.
        
        Returns:
            bool: True if active, False otherwise
        """
        return self.active
    
    def create_trackbars(self, window_name, trackbar_manager):
        """
        Create trackbars for panorama parameters.
        
        Args:
            window_name (str): Name of the window
            trackbar_manager: Trackbar manager instance
        """
        trackbar_manager.create_trackbar('Max Frames', window_name, self.max_frames, 20)
        trackbar_manager.create_trackbar('Reproj Thresh', window_name, int(self.reproj_threshold), 10)
    
    def update_from_trackbars(self, window_name):
        """
        Update panorama parameters from trackbars.
        
        Args:
            window_name (str): Name of the window
        """
        if not self.active:
            return
        
        self.max_frames = max(2, cv.getTrackbarPos('Max Frames', window_name))
        self.reproj_threshold = max(1, cv.getTrackbarPos('Reproj Thresh', window_name))
    
    def start_capture(self):
        """Start capturing frames for panorama."""
        if not self.active:
            return
        
        self.capturing = True
        self.frames = []
        self.panorama = None
        self.current_frame_count = 0
        cv.namedWindow(self.preview_window_name, cv.WINDOW_AUTOSIZE)
        print("Panorama capture started. Press SPACE to capture each frame, S to stitch, R to reset.")
    
    def stop_capture(self):
        """Stop capturing and create panorama."""
        if not self.capturing:
            return
        
        self.capturing = False
        cv.destroyWindow(self.preview_window_name)
        if len(self.frames) >= 2:
            print(f"Stitching {len(self.frames)} frames...")
            self.create_manual_panorama()
        else:
            print("Need at least 2 frames to create panorama.")
    
    def toggle_capture(self):
        """Toggle between start and stop capture."""
        if self.capturing:
            self.stop_capture()
        else:
            self.start_capture()
    
    def process_frame(self, frame):
        """
        Process frame for panorama capture with overlays and instructions.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with status overlay
        """
        display_frame = frame.copy()
        
        if not self.active:
            return display_frame
        
        # Store current frame for manual capture
        self.current_frame = frame.copy()
        
        # Add progress bar and instructions overlay
        self.add_status_overlay(display_frame)
        
        # Always show instructions when in panorama mode
        instructions = [
            "Panorama Mode Active",
            "[SPACE] Capture frame",
            "[S] Stitch panorama",
            "[R] Reset panorama"
        ]
        for i, text in enumerate(instructions):
            cv.putText(display_frame, text, (400, 26 + i*30),
                        cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
        
        # Auto-reset if the panorama window was closed
        if self.panorama is not None:
            try:
                if cv.getWindowProperty(self.panorama_window_name, cv.WND_PROP_VISIBLE) < 1:
                    self.reset()
            except:
                pass
        
        # If a panorama exists, show it in a separate window
        if self.panorama is not None:
            self.show_panorama()
        
        return display_frame



    def capture_frame(self, frame):
        """
        Capture a frame for panorama creation.
        
        Args:
            frame (numpy.ndarray): Frame to capture
        """
        if len(self.frames) >= self.max_frames:
            print(f"Maximum frames ({self.max_frames}) reached.")
            return
        
        # Store frame
        frame_copy = frame.copy()
        self.frames.append(frame_copy)
        self.current_frame_count += 1
        
        # Show preview of captured frame
        preview_frame = cv.resize(frame_copy, (300, 200))  # Small preview
        cv.imshow(self.preview_window_name, preview_frame)
        
        print(f"Captured frame {self.current_frame_count}/{self.max_frames}")
    
    def create_manual_panorama(self):
        """Create panorama using manual feature matching."""
        if len(self.frames) < 2:
            print("Need at least 2 frames for panorama creation.")
            return
        
        try:
            print("Stitching frames manually...")
            
            # Start with the first frame
            result = self.frames[0].copy()
            
            # Stitch each subsequent frame
            for i in range(1, len(self.frames)):
                result = self.stitch_pair(result, self.frames[i])
                if result is None:
                    print(f"Failed to stitch frame {i+1}")
                    return
            
            self.panorama = result
            print("Panorama created successfully!")
            self.show_panorama()
            
        except Exception as e:
            print(f"Manual panorama creation failed: {e}")
    
    def stitch_pair(self, img1, img2):
        """
        Stitch two images together.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            
        Returns:
            numpy.ndarray or None: Stitched image or None if failed
        """
        # Convert to grayscale for feature detection
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        # Detect features
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            print("Insufficient features detected")
            return None
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            print("Insufficient good matches found")
            return None
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        homography, mask = cv.findHomography(
            dst_pts, src_pts, 
            cv.RANSAC, 
            self.reproj_threshold
        )
        
        if homography is None:
            print("Homography calculation failed")
            return None
        
        # Warp and stitch images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners of second image
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_warped = cv.perspectiveTransform(corners2, homography)
        
        # Calculate output size
        all_corners = np.concatenate([
            [[0, 0], [w1, 0], [w1, h1], [0, h1]],
            corners2_warped.reshape(-1, 2)
        ])
        
        x_min, y_min = np.min(all_corners, axis=0).astype(int)
        x_max, y_max = np.max(all_corners, axis=0).astype(int)
        
        # Adjust homography for translation
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        homography_adjusted = translation @ homography
        
        # Create output canvas
        output_width = x_max - x_min
        output_height = y_max - y_min
        result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Warp second image
        warped = cv.warpPerspective(img2, homography_adjusted, (output_width, output_height))
        
        # Place first image
        y_offset = -y_min
        x_offset = -x_min
        result[y_offset:y_offset+h1, x_offset:x_offset+w1] = img1
        
        # Improved blending
        warped_mask = (warped > 0).any(axis=2)
        result_mask = (result > 0).any(axis=2)
        
        # Non-overlap part of warped
        non_overlap = warped_mask & ~result_mask
        result[non_overlap] = warped[non_overlap]
        
        # Overlap
        overlap = warped_mask & result_mask
        if overlap.any():
            # Simple linear blend with alpha=0.5
            alpha = 0.5
            result[overlap] = (alpha * warped[overlap] + (1 - alpha) * result[overlap]).astype(np.uint8)
            
            # Additionally, apply a light blur to the overlap region to smooth any remaining marks
            overlap_indices = np.where(overlap)
            if len(overlap_indices[0]) > 0:
                min_y, max_y = np.min(overlap_indices[0]), np.max(overlap_indices[0])
                min_x, max_x = np.min(overlap_indices[1]), np.max(overlap_indices[1])
                # Extract overlap region
                overlap_region = result[min_y:max_y+1, min_x:max_x+1]
                # Blur it
                blurred = cv.GaussianBlur(overlap_region, (5, 5), 0)
                # Put back
                result[min_y:max_y+1, min_x:max_x+1] = blurred
        
        return result
    
    def show_panorama(self):
        """Display the created panorama in a separate window."""
        if self.panorama is None:
            return
        
        # Resize panorama for display if too large
        display_panorama = self.panorama.copy()
        height, width = display_panorama.shape[:2]
        
        max_display_width = 1200
        if width > max_display_width:
            scale = max_display_width / width
            new_width = max_display_width
            new_height = int(height * scale)
            display_panorama = cv.resize(display_panorama, (new_width, new_height))
        
        cv.imshow(self.panorama_window_name, display_panorama)
    
    def save_panorama(self, filename=None):
        """
        Save the current panorama to file.
        
        Args:
            filename (str, optional): Filename to save to
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if self.panorama is None:
            print("No panorama to save")
            return False
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"panorama_{timestamp}.jpg"
        
        try:
            cv.imwrite(filename, self.panorama)
            print(f"Panorama saved as {filename}")
            return True
        except Exception as e:
            print(f"Error saving panorama: {e}")
            return False
    
    def add_status_overlay(self, frame):
        """
        Add progress bar and text overlay to the frame.
        
        Args:
            frame (numpy.ndarray): Frame to add overlay to
        """
        if not self.capturing:
            return
        
        # Progress bar
        bar_width = 200
        bar_height = 20
        progress = min(self.current_frame_count / self.max_frames, 1.0)
        filled_width = int(bar_width * progress)
        
        # Draw progress bar background
        cv.rectangle(frame, (15, 10), (15 + bar_width, 10 + bar_height), (50, 50, 50), -1)
        # Draw filled portion
        cv.rectangle(frame, (15, 10), (15 + filled_width, 10 + bar_height), (0, 255, 0), -1)
        
        # Add text instructions
        cv.putText(frame, f"Frames: {self.current_frame_count}/{self.max_frames}", (15, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.current_frame_count >= self.max_frames:
            cv.putText(frame, "Max frames reached! Press S to stitch", (15, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def handle_key(self, key):
        """
        Handle keyboard input for panorama mode.
        
        Args:
            key (int): Key code
            
        Returns:
            bool: True if key was handled, False otherwise
        """
        if not self.active:
            return False
        
        if key == ord(' '):  # Space - capture frame or start capture
            if self.capturing and self.current_frame is not None:
                self.capture_frame(self.current_frame)
            else:
                self.toggle_capture()
            return True
        elif key == ord('s'):  # Stitch panorama
            self.stop_capture()
            return True
        elif key == ord('r'):  # Reset
            self.reset()
            return True
        
        return False
    
    def reset(self):
        """Reset panorama state."""
        self.capturing = False
        self.frames = []
        self.panorama = None
        self.current_frame_count = 0
        self.current_frame = None
        
        # Close windows if open
        try:
            cv.destroyWindow(self.panorama_window_name)
            cv.destroyWindow(self.preview_window_name)
            print("Panorama reset.")
        except:
            pass