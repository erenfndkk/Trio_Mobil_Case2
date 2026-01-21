"""
Post-processing module to fix ID switches and merge fragmented tracks
"""
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TrackIDStitcher:
    """
    Post-process tracking results to merge fragmented tracks
    and fix ID switches caused by occlusion
    """

    def __init__(self,
                 max_frame_gap: int = 30,
                 position_threshold: float = 150.0,
                 size_similarity_threshold: float = 0.5):
        """
        Args:
            max_frame_gap: Maximum frames between track end and start to consider merging
            position_threshold: Maximum pixel distance to consider same person
            size_similarity_threshold: Minimum bbox size similarity ratio
        """
        self.max_frame_gap = max_frame_gap
        self.position_threshold = position_threshold
        self.size_similarity_threshold = size_similarity_threshold

    def stitch_tracks(self, tracking_results: List[Dict]) -> List[Dict]:
        """
        Merge fragmented tracks that likely belong to the same person

        Args:
            tracking_results: Original tracking results

        Returns:
            Modified tracking results with merged IDs
        """
        logger.info("Starting track stitching...")

        # Build track segments
        track_segments = self._build_track_segments(tracking_results)

        # Find merge candidates
        merge_map = self._find_merge_candidates(track_segments)

        # Apply ID mapping
        stitched_results = self._apply_id_mapping(tracking_results, merge_map)

        logger.info(f"Stitched {len(merge_map)} track fragments")

        return stitched_results

    def _build_track_segments(self, results: List[Dict]) -> Dict:
        """
        Build track segments with start/end frames and positions

        Returns:
            Dict mapping track_id to segment info
        """
        segments = defaultdict(lambda: {
            'start_frame': float('inf'),
            'end_frame': 0,
            'start_position': None,
            'end_position': None,
            'start_bbox': None,
            'end_bbox': None,
            'frames': []
        })

        for frame_data in results:
            frame_num = frame_data['frame']
            for track in frame_data['tracks']:
                track_id = track['track_id']
                center = track['center']
                bbox = track['bbox']

                seg = segments[track_id]
                seg['frames'].append(frame_num)

                # Update start
                if frame_num < seg['start_frame']:
                    seg['start_frame'] = frame_num
                    seg['start_position'] = center
                    seg['start_bbox'] = bbox

                # Update end
                if frame_num > seg['end_frame']:
                    seg['end_frame'] = frame_num
                    seg['end_position'] = center
                    seg['end_bbox'] = bbox

        return dict(segments)

    def _find_merge_candidates(self, segments: Dict) -> Dict[int, int]:
        """
        Find tracks that should be merged based on spatial-temporal proximity

        Returns:
            Mapping from old_id to new_id (master)
        """
        merge_map = {}
        track_ids = sorted(segments.keys())

        for i, track_id_a in enumerate(track_ids):
            seg_a = segments[track_id_a]

            for track_id_b in track_ids[i + 1:]:
                seg_b = segments[track_id_b]

                # Check if one track starts shortly after another ends
                should_merge = False

                # Case 1: A ends, then B starts
                if seg_a['end_frame'] < seg_b['start_frame']:
                    frame_gap = seg_b['start_frame'] - seg_a['end_frame']
                    if frame_gap <= self.max_frame_gap:
                        # Check position proximity
                        dist = np.linalg.norm(
                            np.array(seg_a['end_position']) -
                            np.array(seg_b['start_position'])
                        )

                        # Check size similarity
                        size_a = self._get_bbox_size(seg_a['end_bbox'])
                        size_b = self._get_bbox_size(seg_b['start_bbox'])
                        size_ratio = min(size_a, size_b) / max(size_a, size_b)

                        if (dist < self.position_threshold and
                                size_ratio > self.size_similarity_threshold):
                            should_merge = True
                            master_id = track_id_a
                            slave_id = track_id_b

                # Case 2: B ends, then A starts
                elif seg_b['end_frame'] < seg_a['start_frame']:
                    frame_gap = seg_a['start_frame'] - seg_b['end_frame']
                    if frame_gap <= self.max_frame_gap:
                        dist = np.linalg.norm(
                            np.array(seg_b['end_position']) -
                            np.array(seg_a['start_position'])
                        )

                        size_a = self._get_bbox_size(seg_a['start_bbox'])
                        size_b = self._get_bbox_size(seg_b['end_bbox'])
                        size_ratio = min(size_a, size_b) / max(size_a, size_b)

                        if (dist < self.position_threshold and
                                size_ratio > self.size_similarity_threshold):
                            should_merge = True
                            master_id = track_id_b
                            slave_id = track_id_a

                if should_merge:
                    # Avoid creating chains, use transitive closure
                    final_master = self._get_master_id(master_id, merge_map)
                    merge_map[slave_id] = final_master
                    logger.debug(f"Merging track {slave_id} into {final_master}")

        return merge_map

    def _get_master_id(self, track_id: int, merge_map: Dict[int, int]) -> int:
        """Get the ultimate master ID following the chain"""
        while track_id in merge_map:
            track_id = merge_map[track_id]
        return track_id

    def _get_bbox_size(self, bbox: List[float]) -> float:
        """Calculate bbox area"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _apply_id_mapping(self, results: List[Dict], merge_map: Dict[int, int]) -> List[Dict]:
        """
        Apply ID mapping to tracking results
        """
        # Expand merge map with transitive closure
        expanded_map = {}
        for old_id in merge_map:
            expanded_map[old_id] = self._get_master_id(old_id, merge_map)

        # Apply mapping
        stitched_results = []
        for frame_data in results:
            new_tracks = []
            for track in frame_data['tracks']:
                track_id = track['track_id']

                # Replace ID if needed
                if track_id in expanded_map:
                    track['track_id'] = expanded_map[track_id]
                    track['stitched'] = True
                else:
                    track['stitched'] = False

                new_tracks.append(track)

            stitched_results.append({
                **frame_data,
                'tracks': new_tracks
            })

        return stitched_results

    def get_statistics(self, original_results: List[Dict],
                       stitched_results: List[Dict]) -> Dict:
        """
        Compare original and stitched results
        """

        def count_unique_ids(results):
            ids = set()
            for frame_data in results:
                for track in frame_data['tracks']:
                    ids.add(track['track_id'])
            return len(ids)

        original_count = count_unique_ids(original_results)
        stitched_count = count_unique_ids(stitched_results)

        return {
            'original_unique_ids': original_count,
            'stitched_unique_ids': stitched_count,
            'ids_merged': original_count - stitched_count,
            'reduction_percentage': (original_count - stitched_count) / original_count * 100
        }