import numpy as np

class LinearKalman1D:
    def __init__(self, dt=1/30, process_noise=0.05, measurement_noise=2.0):
        """
        A standard 1D Kalman Filter tracking angular position and velocity.
        """
        self.dt = dt
        
        # State Vector: [angle, angular_velocity]^T
        self.x = np.zeros((2, 1))
        
        # State Transition Matrix (F)
        self.F = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        
        # Measurement Matrix (H) - We only measure the angle directly
        self.H = np.array([[1.0, 0.0]])
        
        # Process Noise Covariance (Q) - Uncertainty in model physics
        self.Q = np.array([[0.25 * (self.dt**4), 0.5 * (self.dt**3)],
                           [0.5 * (self.dt**3), self.dt**2]]) * process_noise
        
        # Measurement Noise Covariance (R) - Uncertainty in AI detections
        self.R = np.array([[measurement_noise]])
        
        # Error Covariance Matrix (P)
        self.P = np.eye(2) * 1.0
        
        self.is_initialized = False

    def initialize(self, initial_angle):
        self.x = np.array([[initial_angle], [0.0]])
        self.P = np.eye(2) * 1.0
        self.is_initialized = True

    def update(self, measurement=None):
        """
        Executes the prediction-correction loop. Supports measurement=None
        to gracefully handle temporary asset occlusion profiles.
        """
        if not self.is_initialized:
            if measurement is not None:
                self.initialize(measurement)
            return self.x[0, 0]

        # 1. Prediction Step (A priori state propagation)
        x_pred = self.F @ self.x
        P_pred = (self.F @ self.P @ self.F.T) + self.Q

        # 2. Correction Step (A posteriori measurement update)
        if measurement is not None:
            # Innovation (Measurement Residual)
            # Update within src/kinematics/filter.py -> LinearKalman1D.update()
            y = np.array([[measurement]]) - (self.H @ x_pred)
            S = (self.H @ P_pred @ self.H.T) + self.R
            
            # Fast scalar division bypasses heavy LAPACK matrix inversion loops
            K = P_pred @ self.H.T * (1.0 / S[0, 0])
            
            self.x = x_pred + (K @ y)
            self.P = (np.eye(2) - (K @ self.H)) @ P_pred
            
            #y = np.array([[measurement]]) - (self.H @ x_pred)
            ## Innovation Covariance
            #S = (self.H @ P_pred @ self.H.T) + self.R
            ## Dynamic Kalman Gain
            #K = P_pred @ self.H.T @ np.linalg.inv(S)
            #
            ## Update state estimate and error covariance
            #self.x = x_pred + (K @ y)
            #self.P = (np.eye(2) - (K @ self.H)) @ P_pred
        else:
            # TARGET OCCLUDED: Propagate state strictly using velocity history
            self.x = x_pred
            self.P = P_pred

        return self.x[0, 0]


class SphericalGimbalFilter:
    def __init__(self, dt=1/30, process_noise=0.1, measurement_noise=1.5):
        """
        Combines independent Yaw and Pitch filters to create a virtual,
        stabilized electronic gimbal processing stream.
        """
        self.yaw_filter = LinearKalman1D(dt, process_noise, measurement_noise)
        self.pitch_filter = LinearKalman1D(dt, process_noise, measurement_noise)

    def smooth_trajectory(self, raw_yaw, raw_pitch):
        """
        Filters raw noisy angular metrics, outputting stabilized trajectories.
        """
        smooth_yaw = self.yaw_filter.update(raw_yaw)
        smooth_pitch = self.pitch_filter.update(raw_pitch)
        
        return {
            "yaw": round(smooth_yaw, 3),
            "pitch": round(smooth_pitch, 3)
        }


if __name__ == "__main__":
    print("📈 Verification Phase: Testing Spherical Kalman Filter Processing Pipeline...")
    
    # Instantiate filter tracking context assuming standard 30 FPS playback bounds
    gimbal = SphericalGimbalFilter(dt=1/30, process_noise=0.5, measurement_noise=2.0)
    
    # Simulated input telemetry series: Steady motion corrupted by extreme tracking jitter
    simulated_raw_data = [
        {"yaw": 10.0, "pitch": 5.0},
        {"yaw": 12.1, "pitch": 4.8},  
        {"yaw": 25.4, "pitch": 9.2},  # EXTREME AI JITTER FLIP (e.g. spray/snow interference)
        {"yaw": 14.2, "pitch": 5.3},  
        {"yaw": None, "pitch": None}, # FRAME OCCLUSION - Subject hidden behind an obstacle
        {"yaw": 18.5, "pitch": 5.9}   
    ]
    
    print("\nExecuting sequential data filter loop pass:")
    for idx, frame_data in enumerate(simulated_raw_data):
        raw_y, raw_p = frame_data["yaw"], frame_data["pitch"]
        smooth_coords = gimbal.smooth_trajectory(raw_y, raw_p)
        
        status = "NORMAL" if raw_y is not None else "🚫 OCCLUDED (Predictive Projection)"
        print(f"Frame [{idx}]: Status: {status:<30} | Raw: ({str(raw_y):<5}, {str(raw_p):<5}) -> Smooth Gimbal: ({smooth_coords['yaw']}, {smooth_coords['pitch']})")