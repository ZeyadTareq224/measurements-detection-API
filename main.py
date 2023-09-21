import mediapipe as mp
import cv2
import numpy as np

class MeasurementDetector:
    def __init__(self, image_path):
        self.mp_pose = mp.solutions.pose
        self.image_path = image_path

    def _read_image(self) -> any:
        # Read the image using OpenCV
        image = cv2.imread(self.image_path)
        return image
    
    def _convert_image_to_rgb_format(self) -> any:
        image = self._read_image()

        # Convert the image to RGB format (MediaPipe uses RGB images)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def _get_pose_landmarks(self) -> any:
        image_rgb = self._convert_image_to_rgb_format()

        with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(image_rgb)
        return results
    
    def calculate_real_measurements(self) -> dict:
        image = self._read_image()
        results = self._get_pose_landmarks()
        if results.pose_landmarks:
    
            # Calculate distances
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_ankle = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            left_hand = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_hand = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Convert landmarks to pixel coordinates
            image_height, image_width, _ = image.shape
            left_shoulder_coords = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
            right_shoulder_coords = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
            left_hip_coords = (int(left_hip.x * image_width), int(left_hip.y * image_height))
            right_hip_coords = (int(right_hip.x * image_width), int(right_hip.y * image_height))
            left_ankle_coords = (int(left_ankle.x * image_width), int(left_ankle.y * image_height))
            left_hand_coords = (int(left_hand.x * image_width), int(left_hand.y * image_height))
            right_hand_coords = (int(right_hand.x * image_width), int(right_hand.y * image_height))
            
            # Calculate real distances
            distance_pixels = np.linalg.norm(np.array(left_hand_coords) - np.array(right_hand_coords))
            ruler_length_cm = 100  
            pixels_per_cm = distance_pixels / ruler_length_cm
            distance_shoulder_to_shoulder = round(np.linalg.norm(np.array(left_shoulder_coords) - np.array(right_shoulder_coords)) / pixels_per_cm, 2)
            distance_hip_to_hip = round(np.linalg.norm(np.array(left_hip_coords) - np.array(right_hip_coords)) / pixels_per_cm, 2)
            distance_hip_to_ankle = round(np.linalg.norm(np.array(left_hip_coords) - np.array(left_ankle_coords)) / pixels_per_cm, 2)
            distance_shoulder_to_hand = round(np.linalg.norm(np.array(left_shoulder_coords) - np.array(left_hand_coords)) / pixels_per_cm, 2)
            distance_between_hands = round(np.linalg.norm(np.array(left_hand_coords) - np.array(right_hand_coords)) / pixels_per_cm, 2)
            
            return {
                'shoulder2shoulder': distance_shoulder_to_shoulder,
                'hip2hip': distance_hip_to_hip,
                'hip2ankle': distance_hip_to_ankle,
                'shoulder2hand': distance_shoulder_to_hand,
                'hand2hand': distance_between_hands,
                }
    

obj = MeasurementDetector('test_img.jpeg')
print(obj.calculate_real_measurements())

def get_measurements():
    # Initialize the MediaPipe Pose module
    mp_pose = mp.solutions.pose

    # Load a sample image for demonstration
    image_path = 'test_img.jpeg'  # Replace with your image path

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to RGB format (MediaPipe uses RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

    # Draw landmarks on the image
    if results.pose_landmarks:
        
        # Calculate distances
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Convert landmarks to pixel coordinates
        image_height, image_width, _ = image.shape
        left_shoulder_coords = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
        right_shoulder_coords = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
        left_hip_coords = (int(left_hip.x * image_width), int(left_hip.y * image_height))
        right_hip_coords = (int(right_hip.x * image_width), int(right_hip.y * image_height))
        left_ankle_coords = (int(left_ankle.x * image_width), int(left_ankle.y * image_height))
        left_hand_coords = (int(left_hand.x * image_width), int(left_hand.y * image_height))
        right_hand_coords = (int(right_hand.x * image_width), int(right_hand.y * image_height))
        
        # Calculate real distances
        distance_pixels = np.linalg.norm(np.array(left_hand_coords) - np.array(right_hand_coords))
        ruler_length_cm = 100  
        pixels_per_cm = distance_pixels / ruler_length_cm
        distance_shoulder_to_shoulder = round(np.linalg.norm(np.array(left_shoulder_coords) - np.array(right_shoulder_coords)) / pixels_per_cm, 2)
        distance_hip_to_hip = round(np.linalg.norm(np.array(left_hip_coords) - np.array(right_hip_coords)) / pixels_per_cm, 2)
        distance_hip_to_ankle = round(np.linalg.norm(np.array(left_hip_coords) - np.array(left_ankle_coords)) / pixels_per_cm, 2)
        distance_shoulder_to_hand = round(np.linalg.norm(np.array(left_shoulder_coords) - np.array(left_hand_coords)) / pixels_per_cm, 2)
        distance_between_hands = round(np.linalg.norm(np.array(left_hand_coords) - np.array(right_hand_coords)) / pixels_per_cm, 2)
        
        return {
            'shoulder2shoulder': distance_shoulder_to_shoulder,
            'hip2hip': distance_hip_to_hip,
            'hip2ankle': distance_hip_to_ankle,
            'shoulder2hand': distance_shoulder_to_hand,
            'hand2hand': distance_between_hands,
            }


