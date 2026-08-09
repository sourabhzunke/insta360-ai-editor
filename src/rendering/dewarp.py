import cv2
import numpy as np

class FisheyeDewarpEngine:
    def __init__(self, output_w=1280, output_h=720, fov_deg=90.0):
        """
        Projects raw Insta360 circular dual-fisheye images into 16:9 perspective viewports.
        Uses per-pixel ray mapping to handle transitions across front and back lenses.
        """
        self.out_w = output_w
        self.out_h = output_h
        self.fov = np.radians(fov_deg)

        # Precompute static 3D perspective camera rays
        f = 0.5 * self.out_w / np.tan(self.fov / 2.0)
        x = np.linspace(-self.out_w / 2, self.out_w / 2 - 1, self.out_w)
        y = np.linspace(-self.out_h / 2, self.out_h / 2 - 1, self.out_h)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx) * f

        directions = np.stack([xx, yy, zz], axis=-1)
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        self.base_rays = (directions / norms).astype(np.float32)

    def _build_ray_map(self, yaw_deg, pitch_deg, roll_deg, src_w, src_h):
        rad_y, rad_p, rad_r = np.radians([yaw_deg, pitch_deg, roll_deg])
        
        Ry = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]], dtype=np.float32)
        Rp = np.array([[1, 0, 0], [0, np.cos(rad_p), -np.sin(rad_p)], [0, np.sin(rad_p), np.cos(rad_p)]], dtype=np.float32)
        Rr = np.array([[np.cos(rad_r), -np.sin(rad_r), 0], [np.sin(rad_r), np.cos(rad_r), 0], [0, 0, 1]], dtype=np.float32)
        
        R = Ry @ Rp @ Rr
        rotated_rays = np.dot(self.base_rays, R.T)

        rx = rotated_rays[..., 0]
        ry = rotated_rays[..., 1]
        rz = rotated_rays[..., 2]

        radius_pixels = src_h / 2.0

        # 🚀 PER-PIXEL LENS MAPPING: Determines front vs back lens for every ray individually
        is_front = (rz >= 0)

        # Front Lens Calculations (Z >= 0)
        theta_front = np.arccos(np.clip(rz, -1.0, 1.0))
        phi_front = np.arctan2(ry, rx)
        r_front = theta_front / np.pi
        map_x_front = (src_w * 0.25) + r_front * np.cos(phi_front) * radius_pixels
        map_y_front = (src_h * 0.5) + r_front * np.sin(phi_front) * radius_pixels

        # Back Lens Calculations (Z < 0)
        theta_back = np.arccos(np.clip(-rz, -1.0, 1.0))
        phi_back = np.arctan2(ry, -rx)
        r_back = theta_back / np.pi
        map_x_back = (src_w * 0.75) + r_back * np.cos(phi_back) * radius_pixels
        map_y_back = (src_h * 0.5) + r_back * np.sin(phi_back) * radius_pixels

        # Combine maps seamlessly per pixel
        map_x = np.where(is_front, map_x_front, map_x_back).astype(np.float32)
        map_y = np.where(is_front, map_y_front, map_y_back).astype(np.float32)

        return map_x, map_y

    def extract_perspective_viewport(self, raw_frame, yaw_deg, pitch_deg, roll_deg=0.0):
        src_h, src_w = raw_frame.shape[:2]
        map_x, map_y = self._build_ray_map(yaw_deg, pitch_deg, roll_deg, src_w, src_h)
        return cv2.remap(raw_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)