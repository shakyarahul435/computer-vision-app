import os 
import numpy as np 
import cv2 as cv

class AugmentedRealityHandler:
    """Handles augmented reality features."""
    
    def __init__(self, calibration_file='calibration_data.npz', model_path='models/trex_model.obj', marker_length=0.12, model_scale_factor=0.0004, rotate_model=True):
        """Initialize AR state."""
        self.active = False
        self.calibration_file = calibration_file
        self.model_path = model_path
        self.marker_length = marker_length  # Marker size in meters
        self.mtx = np.eye(3)
        self.dist = np.zeros((1, 5))
        self.aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
        self.aruco_params = cv.aruco.DetectorParameters()
        self.aruco_detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.model_vertices = None
        self.model_faces = None
        self._load_calibration()
        self.pinhole_params = {'fx': self.mtx[0, 0], 'fy': self.mtx[1, 1], 'cx': self.mtx[0, 2], 'cy': self.mtx[1, 2]}
        self._load_model(model_scale_factor, rotate_model)

    def set_active(self, active):
        """
        Set the active state of AR.
        
        Args:
            active (bool): Whether AR is active
        """
        self.active = active
    
    def is_active(self):
        """Check if AR is active."""
        return self.active

    def _load_calibration(self):
        if os.path.exists(self.calibration_file):
            with np.load(self.calibration_file) as X:
                self.mtx, self.dist = [X[i] for i in ('mtx', 'dist')]
            print("Calibration data loaded.")
        else:
            print(f"WARNING: '{self.calibration_file}' not found. AR and Pinhole modes will not be accurate.")
    
    def _load_model(self, scale_factor=None, rotate=False):
        """Load and preprocess the 3D model."""
        if os.path.exists(self.model_path):
            vertices = []
            faces = []
            with open(self.model_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                                vertices.append(vertex)
                            except ValueError:
                                continue
                    elif line.startswith('f '):
                        parts = line.split()[1:]
                        try:
                            face = []
                            for p in parts:
                                idx = int(p.split('/')[0]) - 1
                                face.append(idx)
                            if len(face) >= 3 and all(0 <= idx < len(vertices) for idx in face):
                                faces.append(face)
                        except (ValueError, IndexError):
                            continue
            
            if vertices and faces:
                self.model_vertices = np.array(vertices, dtype=np.float32)
                self.model_faces = faces
                
                # Center the model by subtracting centroid
                centroid = np.mean(self.model_vertices, axis=0)
                self.model_vertices -= centroid
                
                # Rotation to orient upright (e.g., if model is lying on side)
                if rotate:
                    rotation_x = np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]
                    ])

                    self.model_vertices = np.dot(self.model_vertices, rotation_x.T)

                # Shift so bottom is on the plane (Z=0)
                bb_min = np.min(self.model_vertices, axis=0)
                self.model_vertices[:, 2] -= bb_min[2]
                
                # Auto-scale if no scale_factor provided
                if scale_factor is None:
                    # Compute bounding box max dimension
                    bb_min = np.min(self.model_vertices, axis=0)
                    bb_max = np.max(self.model_vertices, axis=0)
                    bb_size = bb_max - bb_min
                    max_dim = np.max(bb_size)
                    if max_dim > 0:
                        scale_factor = self.marker_length / max_dim * 2.0  # Increased to 200% for larger size
                    else:
                        scale_factor = 1.0
                print(f"Auto scale factor: {scale_factor}")
                # Apply scaling
                self.model_vertices *= scale_factor
                
                print(f"Model loaded, oriented, and scaled by factor {scale_factor}.")
            else:
                print("No valid vertices or faces found in model.")
        else:
            print(f"Model file '{self.model_path}' not found.")
    
    def draw_3d_model(self, frame, window_name):
        # Detect ArUco markers
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        
        if ids is not None and len(ids) > 0:
            # Assume first marker
            marker_corners = corners[0].reshape((4, 2))
            image_points = marker_corners.reshape(-1, 2).astype(np.float32)

            objp = np.array([
                [-self.marker_length/2,  self.marker_length/2, 0],  # top-left
                [ self.marker_length/2,  self.marker_length/2, 0],  # top-right
                [ self.marker_length/2, -self.marker_length/2, 0],  # bottom-right
                [-self.marker_length/2, -self.marker_length/2, 0],  # bottom-left
            ], dtype=np.float32)
            
            success, rvec, tvec = cv.solvePnP(
                objp, image_points, self.mtx, self.dist
            )

            if success and self.model_vertices is not None and self.model_faces:
                try:
                    # Get rotation matrix from rvec
                    R, _ = cv.Rodrigues(rvec)
                    
                    # Transform vertices to camera space for depth calculation
                    vertices_cam = np.dot(self.model_vertices, R.T) + tvec.T
                    
                    # Compute depth (average Z) for each face
                    face_depths = []
                    for i, face in enumerate(self.model_faces):
                        if len(face) >= 3:
                            avg_z = np.mean(vertices_cam[face, 2])
                            face_depths.append((i, avg_z))
                    
                    # Sort faces by depth (farthest to nearest)
                    face_depths.sort(key=lambda x: x[1], reverse=True)
                    
                    # Project all vertices
                    imgpts, _ = cv.projectPoints(self.model_vertices, rvec, tvec, self.mtx, self.dist)
                    imgpts = np.int32(imgpts).reshape(-1, 2)
                    
                    # Get frame dimensions for clipping
                    height, width = frame.shape[:2]
                    
                    # Draw sorted faces
                    for i, _ in face_depths:
                        face = self.model_faces[i]
                        if len(face) >= 3:
                            pts = imgpts[face]
                            # Clip points to frame bounds
                            pts = np.clip(pts, [0, 0], [width - 1, height - 1])
                            # Fill with solid green (assuming convex faces)
                            cv.fillConvexPoly(frame, pts, (0, 255, 0))  # Solid green
                except Exception as e:
                    print(f"Error during projection or drawing: {e}")
            else:
                print("Model not loaded.")
        
        return frame
