"""
Visualization module for creating overlay videos with tracking and counting info
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TrackingVisualizer:
    """
    Create visualization overlays for tracking and counting
    """

    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        """
        Args:
            line_start: (x, y) start point of counting line
            line_end: (x, y) end point of counting line
        """
        self.line_start = line_start
        self.line_end = line_end

        # Colors
        self.line_color = (0, 255, 255)  # Yellow
        self.bbox_color = (0, 255, 0)  # Green
        self.text_color = (255, 255, 255)  # White
        self.crossing_in_color = (0, 255, 0)  # Green
        self.crossing_out_color = (0, 0, 255)  # Red

    def create_video(self,
                     image_folder: Path,
                     tracking_results: List[Dict],
                     events: List[Dict],
                     output_path: Path,
                     fps: int = 25,
                     max_frames: int = None,
                     draw_trails: bool = True) -> None:
        """
        Create visualization video with tracking overlay

        Args:
            image_folder: Path to sequence images
            tracking_results: Tracking results from tracker
            events: Crossing events from counter
            output_path: Output video path
            fps: Frames per second
            max_frames: Maximum frames to process (None for all)
            draw_trails: Whether to draw track trails
        """
        # Get sorted images
        image_files = sorted(list(image_folder.glob('*.jpg')))

        if max_frames:
            image_files = image_files[:max_frames]

        # Get frame size
        first_frame = cv2.imread(str(image_files[0]))
        height, width = first_frame.shape[:2]

        # Create video writer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Build event lookup by frame
        events_by_frame = {}
        for event in events:
            frame_num = event['frame']
            if frame_num not in events_by_frame:
                events_by_frame[frame_num] = []
            events_by_frame[frame_num].append(event)

        # Track trails
        track_trails = {}  # track_id -> list of centers

        # Current counts
        count_in = 0
        count_out = 0

        logger.info(f"Creating video with {len(image_files)} frames...")

        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            frame_num = idx + 1

            # Get tracking data for this frame
            if idx < len(tracking_results):
                tracks = tracking_results[idx]['tracks']
            else:
                tracks = []

            # Draw counting line
            cv2.line(frame, self.line_start, self.line_end,
                    self.line_color, 3)

            # Update trails and draw tracks
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                center = track['center']

                # Update trail
                if track_id not in track_trails:
                    track_trails[track_id] = []
                track_trails[track_id].append(center)

                # Keep only recent positions
                if len(track_trails[track_id]) > 30:
                    track_trails[track_id].pop(0)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, 2)

                # Draw track ID
                label = f"ID:{track_id}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)

                # Draw center point
                cx, cy = map(int, center)
                cv2.circle(frame, (cx, cy), 4, self.bbox_color, -1)

            # Draw trails
            if draw_trails:
                for track_id, trail in track_trails.items():
                    if len(trail) > 1:
                        points = np.array(trail, dtype=np.int32)
                        cv2.polylines(frame, [points], False, (255, 0, 255), 2)

            # Check for crossing events in this frame
            if frame_num in events_by_frame:
                for event in events_by_frame[frame_num]:
                    if event['direction'] == 'IN':
                        count_in += 1
                        color = self.crossing_in_color
                    else:
                        count_out += 1
                        color = self.crossing_out_color

                    # Flash effect: draw circle at crossing position
                    pos = event['position']
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), 20, color, 3)

                    # Show direction arrow near crossing
                    arrow_text = "IN" if event['direction'] == 'IN' else "OUT"
                    cv2.putText(frame, arrow_text,
                               (int(pos[0]) + 25, int(pos[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw counter info
            self._draw_counter_panel(frame, frame_num, count_in, count_out, len(tracks))

            # Write frame
            out.write(frame)

            if (frame_num) % 100 == 0:
                logger.info(f"Processed {frame_num}/{len(image_files)} frames")

        out.release()
        logger.info(f"Video saved to {output_path}")

    def _draw_counter_panel(self, frame: np.ndarray,
                           frame_num: int,
                           count_in: int,
                           count_out: int,
                           active_tracks: int) -> None:
        """Draw information panel on frame"""
        height, width = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text information
        y_offset = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        texts = [
            f"Frame: {frame_num}",
            f"Active Tracks: {active_tracks}",
            f"IN: {count_in}",
            f"OUT: {count_out}",
            f"Net: {count_in - count_out}"
        ]

        for i, text in enumerate(texts):
            y_pos = y_offset + i * 25
            cv2.putText(frame, text, (20, y_pos),
                       font, font_scale, self.text_color, thickness)

    def create_summary_image(self,
                           summary: Dict,
                           events: List[Dict],
                           output_path: Path) -> None:
        """
        Create a summary visualization image
        """
        # Create blank canvas
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255

        # Title
        cv2.putText(img, "Counting Summary", (250, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        # Statistics
        y_offset = 120
        stats_text = [
            f"Total IN: {summary['total_in']}",
            f"Total OUT: {summary['total_out']}",
            f"Net Count: {summary['net']}",
            f"Total Crossings: {summary['total_crossings']}",
            f"Unique Tracks: {summary['unique_tracks']}"
        ]

        for i, text in enumerate(stats_text):
            y_pos = y_offset + i * 50
            cv2.putText(img, text, (100, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        logger.info(f"Summary image saved to {output_path}")