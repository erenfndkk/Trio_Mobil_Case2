"""
Main script to run tracking and counting pipeline with two combinations

Usage:
    python src/run.py

This will run both combinations:
1. YOLO11n + BoT-SORT + ID Stitching
2. YOLO11s + ByteTrack (default YOLO tracker)
"""
import yaml
import json
import logging
from pathlib import Path
import pandas as pd

from tracker_botsort import PersonTrackerBoTSORT
from tracker_bytetrack import PersonTrackerByteTrack
from counter import LineCrossingCounter
from visualizer import TrackingVisualizer
from id_stitcher import TrackIDStitcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_sequence(sequence_name: str,
                     data_root: Path,
                     config: dict,
                     output_root: Path,
                     combination_name: str,
                     tracker_type: str = 'botsort',
                     use_stitching: bool = True,
                     max_frames: int = None) -> dict:
    """
    Process one MOT sequence with specified tracker
    
    Args:
        sequence_name: Name of sequence (e.g., 'MOT17-04')
        data_root: Root path to MOT17 data
        config: Configuration dictionary
        output_root: Root path for outputs
        combination_name: Name of the combination (e.g., 'nano+botsort')
        tracker_type: 'botsort' or 'bytetrack'
        use_stitching: Whether to apply ID stitching (only for botsort)
        max_frames: Maximum frames to process (None for all)
    
    Returns:
        Dictionary with results summary
    """
    logger.info(f"=" * 80)
    logger.info(f"Processing: {sequence_name} | Tracker: {tracker_type.upper()} | Stitching: {use_stitching}")
    logger.info(f"=" * 80)
    
    # Paths
    sequence_path = data_root / sequence_name / 'img1'
    if not sequence_path.exists():
        raise ValueError(f"Sequence path not found: {sequence_path}")
    
    # Create output folder based on combination
    output_seq = output_root / combination_name / sequence_name
    output_seq.mkdir(parents=True, exist_ok=True)
    
    # Get sequence config
    seq_config = config['sequences'][sequence_name]
    line_coords = seq_config['line']
    line_start = (line_coords[0], line_coords[1])
    line_end = (line_coords[2], line_coords[3])
    
    buffer_config = config['buffer']
    
    # Step 1: Tracking with selected tracker
    logger.info(f"Step 1: Running tracking with {tracker_type.upper()}...")
    
    # Define tracks.txt path
    tracks_file = output_seq / 'tracks.txt'
    
    if tracker_type == 'botsort':
        tracker = PersonTrackerBoTSORT(model_name='yolo11n.pt', conf_threshold=0.3)
    else:  # bytetrack
        tracker = PersonTrackerByteTrack(model_name='yolo11s.pt', conf_threshold=0.3)
    
    tracking_results = tracker.track_sequence(
        sequence_path,
        output_tracks=tracks_file  # Save tracks.txt
    )
    
    # Limit frames if specified
    if max_frames:
        tracking_results = tracking_results[:max_frames]
        logger.info(f"Limited to {max_frames} frames")
    
    # Step 1.5: ID Stitching (Post-processing) - only for botsort
    stitching_stats = None
    if use_stitching and tracker_type == 'botsort':
        logger.info("Step 1.5: Applying ID stitching post-processing...")
        stitcher = TrackIDStitcher(
            max_frame_gap=30,
            position_threshold=150.0,
            size_similarity_threshold=0.5
        )
        
        tracking_results_stitched = stitcher.stitch_tracks(tracking_results)
        stitching_stats = stitcher.get_statistics(tracking_results, tracking_results_stitched)
        
        logger.info(f"Stitching stats: {stitching_stats}")
        
        # Use stitched results for counting
        tracking_results = tracking_results_stitched
        
        # Re-save tracks.txt with stitched IDs
        tracker._save_tracks_mot_format(tracking_results, tracks_file)
    
    # Step 2: Counting
    logger.info("Step 2: Running line crossing detection...")
    counter = LineCrossingCounter(
        line_start=line_start,
        line_end=line_end,
        min_frames_between_crossings=buffer_config['min_frames_between_crossings'],
        position_history_length=buffer_config['position_history_length']
    )
    
    # Process all frames
    for frame_data in tracking_results:
        frame_num = frame_data['frame']
        tracks = frame_data['tracks']
        counter.process_frame(frame_num, tracks)
    
    # Get results
    summary = counter.get_summary()
    events = counter.get_events()
    
    logger.info(f"Counting complete: {summary}")
    
    # Save events
    events_path = output_seq / 'events.json'
    with open(events_path, 'w') as f:
        json.dump({
            'summary': summary,
            'events': events,
            'config': seq_config,
            'tracker': tracker_type,
            'stitching_enabled': use_stitching,
            'stitching_stats': stitching_stats
        }, f, indent=2)
    logger.info(f"Events saved to {events_path}")
    
    # Create events CSV
    if events:
        events_df = pd.DataFrame(events)
        events_df.to_csv(output_seq / 'events.csv', index=False)
    
    # Step 3: Visualization
    logger.info("Step 3: Creating visualization video...")
    visualizer = TrackingVisualizer(line_start, line_end)
    
    video_path = output_seq / f'{sequence_name}_demo.mp4'
    visualizer.create_video(
        image_folder=sequence_path,
        tracking_results=tracking_results,
        events=events,
        output_path=video_path,
        fps=25,
        max_frames=max_frames,
        draw_trails=False
    )
    
    # Create summary image
    summary_img_path = output_seq / 'summary.png'
    visualizer.create_summary_image(summary, events, summary_img_path)
    
    logger.info(f"Sequence {sequence_name} with {tracker_type} processing complete!")
    logger.info(f"Outputs saved to: {output_seq}")
    
    return {
        'sequence': sequence_name,
        'tracker': tracker_type,
        'stitching_enabled': use_stitching,
        'summary': summary,
        'num_frames': len(tracking_results),
        'output_path': str(output_seq),
        'stitching_stats': stitching_stats,
        'tracks_file': str(tracks_file)
    }


def main():
    # Get project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    
    # Paths relative to project root
    data_root = project_root / 'data' / 'MOT17'
    config_path = project_root / 'configs' / 'lines.yaml'
    output_root = project_root / 'outputs'
    
    # Sequences to process
    sequences = ['MOT17-04', 'MOT17-09', 'MOT17-13']
    
    # Combinations to run
    combinations = [
        {'tracker': 'botsort', 'stitching': True, 'name': 'BoT-SORT + ID Stitching', 'output_name': 'nano+botsort'},
        {'tracker': 'bytetrack', 'stitching': False, 'name': 'ByteTrack (YOLO default)', 'output_name': 'small+bytetrack'}
    ]
    
    logger.info("=" * 80)
    logger.info("MOT17 Tracking and Counting Pipeline - Multiple Combinations")
    logger.info("=" * 80)
    
    # Load config
    config = load_config(config_path)
    
    # Store all results
    all_results = []
    
    # Process each combination
    for combo in combinations:
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING COMBINATION: {combo['name']}")
        logger.info("=" * 80 + "\n")
        
        combo_results = []
        
        # Process each sequence with this combination
        for seq_name in sequences:
            try:
                result = process_sequence(
                    sequence_name=seq_name,
                    data_root=data_root,
                    config=config,
                    output_root=output_root,
                    combination_name=combo['output_name'],
                    tracker_type=combo['tracker'],
                    use_stitching=combo['stitching'],
                    max_frames=None
                )
                combo_results.append(result)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {seq_name} with {combo['name']}: {e}", exc_info=True)
        
        # Save combination summary
        combo_summary_path = output_root / combo['output_name'] / 'combination_summary.json'
        combo_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combo_summary_path, 'w') as f:
            json.dump({
                'combination': combo['output_name'],
                'tracker': combo['tracker'],
                'stitching': combo['stitching'],
                'results': combo_results
            }, f, indent=2)
        
        logger.info(f"\n{combo['name']} completed!")
        logger.info(f"Summary saved to: {combo_summary_path}\n")
    
    # Save overall summary
    overall_summary_path = output_root / 'overall_summary.json'
    with open(overall_summary_path, 'w') as f:
        json.dump({
            'combinations': combinations,
            'sequences': sequences,
            'results': all_results
        }, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL COMBINATIONS PROCESSED!")
    logger.info(f"Overall summary saved to: {overall_summary_path}")
    logger.info("=" * 80)
    
    # Print comparative summary table
    print("\n" + "=" * 80)
    print("COMPARATIVE RESULTS SUMMARY")
    print("=" * 80)
    
    for combo in combinations:
        print(f"\n{'='*80}")
        print(f"COMBINATION: {combo['name']}")
        print(f"{'='*80}")
        
        combo_results = [r for r in all_results if r['tracker'] == combo['tracker']]
        
        for result in combo_results:
            print(f"\n  Sequence: {result['sequence']}")
            print(f"    Frames: {result['num_frames']}")
            print(f"    Tracks file: {result['tracks_file']}")
            
            summary = result['summary']
            print(f"    Counting:")
            print(f"      IN:  {summary['total_in']}")
            print(f"      OUT: {summary['total_out']}")
            print(f"      Net: {summary['net']}")
            print(f"      Unique tracks: {summary['unique_tracks']}")
    
    print("\n" + "=" * 80)
    print("Output folder structure:")
    print("  outputs/")
    print("    ├── nano+botsort/")
    print("    │   ├── MOT17-04/")
    print("    │   │   ├── tracks.txt")
    print("    │   │   ├── events.json")
    print("    │   │   └── MOT17-04.mp4")
    print("    │   ├── MOT17-09/")
    print("    │   └── MOT17-13/")
    print("    ├── small+bytetrack/")
    print("    │   ├── MOT17-04/")
    print("    │   ├── MOT17-09/")
    print("    │   └── MOT17-13/")
    print("    └── overall_summary.json")
    print("=" * 80)


if __name__ == '__main__':
    main()