"""
Person tracking using YOLO11n + BoT-SORT tracker
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonTrackerBoTSORT:
    """
    Person detection and tracking using YOLO11n with BoT-SORT tracker
    """
    
    def __init__(self, model_name: str = 'yolo11n.pt', conf_threshold: float = 0.3):
        """
        Initialize tracker
        
        Args:
            model_name: YOLO model to use (yolo11n for speed on CPU)
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        logger.info(f"Loaded model: {model_name} with BoT-SORT tracker")
    
    def track_sequence(self, image_folder: Path, output_tracks: Path = None) -> List[Dict]:
        """
        Track persons through a sequence of images
        
        Args:
            image_folder: Path to folder containing sequence images
            output_tracks: Path to save tracking results in MOT format
            
        Returns:
            List of tracking results per frame
        """
        # Get sorted list of images
        image_files = sorted(list(image_folder.glob('*.jpg')))
        if not image_files:
            raise ValueError(f"No images found in {image_folder}")
        
        logger.info(f"Found {len(image_files)} images in sequence")
        
        all_results = []
        
        for frame_idx, img_path in enumerate(image_files):
            # Read image
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"Could not read {img_path}")
                continue
            
            # Run tracking with BoT-SORT
            results = self.model.track(
                frame,
                persist=True,
                tracker='botsort.yaml',  # Use BoT-SORT tracker
                conf=self.conf_threshold,
                iou=0.5,
                classes=[0],  # 0 is person class in COCO
                verbose=False
            )
            
            # Extract tracking info
            frame_tracks = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    frame_tracks.append({
                        'frame': frame_idx + 1,
                        'track_id': int(track_id),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(center_x), float(center_y)],
                        'confidence': float(conf)
                    })
            
            all_results.append({
                'frame': frame_idx + 1,
                'image_path': str(img_path),
                'tracks': frame_tracks
            })
            
            if (frame_idx + 1) % 50 == 0:
                logger.info(f"Processed {frame_idx + 1}/{len(image_files)} frames")
        
        # ALWAYS save tracks in MOT format
        if output_tracks:
            self._save_tracks_mot_format(all_results, output_tracks)
        
        logger.info(f"Tracking complete: {len(all_results)} frames processed")
        return all_results
    
    def _save_tracks_mot_format(self, results: List[Dict], output_path: Path):
        """
        Save tracking results in MOT challenge format
        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for frame_data in results:
                frame_num = frame_data['frame']
                for track in frame_data['tracks']:
                    x1, y1, x2, y2 = track['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    track_id = track['track_id']
                    conf = track['confidence']
                    
                    # MOT format
                    line = f"{frame_num},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1\n"
                    f.write(line)
        
        logger.info(f"Saved tracks to {output_path}")
    
    def get_frame_shape(self, image_folder: Path) -> Tuple[int, int]:
        """Get the shape (height, width) of frames in the sequence"""
        first_image = next(image_folder.glob('*.jpg'))
        frame = cv2.imread(str(first_image))
        return frame.shape[:2]  # (height, width)