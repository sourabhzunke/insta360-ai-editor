import numpy as np

class SphericalCoordinateTransformer:
    def __init__(self, frame_width=640, frame_height=480):
        """
        Translates raw 2D bounding box coordinate centers into spherical 
        angular trajectories (Yaw/Pitch) matching equirectangular canvas vectors.
        """
        self.W = frame_width
        self.H = frame_height

    def bbox_to_angles(self, bbox):
        """
        Converts a 2D bounding box [x1, y1, x2, y2] into horizontal 
        Yaw and vertical Pitch angles in degrees.
        
        Horizontal Axis: -180 to +180 degrees panorama sweep
        Vertical Axis: -90 to +90 degrees tilt sweep
        """
        if not bbox or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox
        
        # 1. Compute the geometric center point of the target bounding box
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        # 2. Map pixel centers linearly to angular orientation values
        # Normalized coordinates range from -0.5 to +0.5 relative to the sensor center
        norm_x = (x_center / self.W) - 0.5
        norm_y = 0.5 - (y_center / self.H) # Invert Y so up is a positive angle

        # 3. Calculate degrees
        yaw = norm_x * 360.0    # Map to full panorama sweep range
        pitch = norm_y * 180.0  # Map to full up/down tilt range

        return {
            "yaw": round(yaw, 3),
            "pitch": round(pitch, 3),
            "pixel_center": (int(x_center), int(y_center))
        }