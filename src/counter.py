"""
Line crossing counter with anti-double-counting logic
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class LineCrossingCounter:
    """
    Counts people crossing a virtual line with direction detection
    """

    def __init__(self,
                 line_start: Tuple[float, float],
                 line_end: Tuple[float, float],
                 min_frames_between_crossings: int = 30,
                 position_history_length: int = 10):
        """
        Args:
            line_start: (x, y) coordinates of line start
            line_end: (x, y) coordinates of line end
            min_frames_between_crossings: Minimum frames before same ID can cross again
            position_history_length: Number of positions to keep in history per track
        """
        self.line_start = np.array(line_start)
        self.line_end = np.array(line_end)
        self.min_frames_between = min_frames_between_crossings
        self.history_length = position_history_length

        # Track history: track_id -> deque of (frame, center_position)
        self.track_history = defaultdict(lambda: deque(maxlen=position_history_length))

        # Last crossing frame for each track_id
        self.last_crossing_frame = {}

        # Counters
        self.count_in = 0
        self.count_out = 0

        # Event log
        self.events = []

        logger.info(f"Initialized counter with line: {line_start} -> {line_end}")

    def process_frame(self, frame_num: int, tracks: List[Dict]) -> List[Dict]:
        """
        Process one frame and detect line crossings

        Args:
            frame_num: Current frame number
            tracks: List of track dictionaries with 'track_id' and 'center'

        Returns:
            List of crossing events in this frame
        """
        frame_events = []

        for track in tracks:
            track_id = track['track_id']
            center = np.array(track['center'])

            # Add to history
            self.track_history[track_id].append((frame_num, center))

            # Need at least 2 positions to detect crossing
            if len(self.track_history[track_id]) < 2:
                continue

            # Check if enough time passed since last crossing
            if track_id in self.last_crossing_frame:
                frames_since_last = frame_num - self.last_crossing_frame[track_id]
                if frames_since_last < self.min_frames_between:
                    continue

            # Get previous position
            prev_frame, prev_center = self.track_history[track_id][-2]

            # Check if line was crossed
            crossing_info = self._check_line_crossing(prev_center, center)

            if crossing_info is not None:
                direction = crossing_info['direction']

                # Update counters
                if direction == 'IN':
                    self.count_in += 1
                elif direction == 'OUT':
                    self.count_out += 1

                # Record event
                event = {
                    'frame': frame_num,
                    'track_id': track_id,
                    'direction': direction,
                    'position': center.tolist(),
                    'count_in': self.count_in,
                    'count_out': self.count_out
                }
                self.events.append(event)
                frame_events.append(event)

                # Update last crossing frame
                self.last_crossing_frame[track_id] = frame_num

                logger.debug(f"Frame {frame_num}: Track {track_id} crossed {direction}")

        return frame_events

    def _check_line_crossing(self, pos1: np.ndarray, pos2: np.ndarray) -> Dict:
        """
        Check if trajectory from pos1 to pos2 crosses the line
        Uses line segment intersection

        Returns:
            Dict with crossing info or None
        """
        # Check if line segment (pos1, pos2) intersects with counting line
        if self._segments_intersect(pos1, pos2, self.line_start, self.line_end):
            # Determine direction based on which side of the line
            direction = self._get_crossing_direction(pos1, pos2)
            return {'direction': direction}

        return None

    def _segments_intersect(self, p1: np.ndarray, p2: np.ndarray,
                           p3: np.ndarray, p4: np.ndarray) -> bool:
        """
        Check if line segment p1-p2 intersects with line segment p3-p4
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _get_crossing_direction(self, pos1: np.ndarray, pos2: np.ndarray) -> str:
        """
        Determine crossing direction using cross product
        Positive cross product = left to right / bottom to top = IN
        Negative = OUT
        """
        # Line vector
        line_vec = self.line_end - self.line_start

        # Movement vector
        movement_vec = pos2 - pos1

        # Cross product (2D)
        cross = line_vec[0] * movement_vec[1] - line_vec[1] * movement_vec[0]

        return 'IN' if cross > 0 else 'OUT'

    def get_summary(self) -> Dict:
        """Get counting summary"""
        return {
            'total_in': self.count_in,
            'total_out': self.count_out,
            'net': self.count_in - self.count_out,
            'total_crossings': len(self.events),
            'unique_tracks': len(set(e['track_id'] for e in self.events))
        }

    def get_events(self) -> List[Dict]:
        """Get all crossing events"""
        return self.events