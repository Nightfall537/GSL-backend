"""
Dataset Packaging Module
Creates class_index.json and dataset_info.json
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from ..utils.logger import setup_logger
from ..config.training_config import OUTPUT_DIR, METADATA_FILE

logger = setup_logger(__name__)


class GSLDatasetPackager:
    """Package GSL dataset with proper validation"""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def create_class_index(self, labels: List[str]) -> Dict[str, int]:
        """
        Create label → index mapping
        
        Args:
            labels: List of gesture labels
        
        Returns:
            Dictionary mapping labels to indices
        """
        unique_labels = sorted(set(labels))
        class_index = {label: idx for idx, label in enumerate(unique_labels)}
        
        logger.info(f"📋 Created class index with {len(class_index)} classes")
        
        return class_index
    
    def save_class_index(self, class_index: Dict[str, int]):
        """Save class index to JSON"""
        output_file = self.output_dir / "class_index.json"
        
        data = {
            **class_index,
            'num_classes': len(class_index)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved class index: {output_file}")
    
    def create_dataset_info(self, landmarks_file: Path, embeddings_file: Path):
        """
        Create comprehensive dataset information file
        
        Args:
            landmarks_file: Path to landmarks JSON
            embeddings_file: Path to embeddings NPY
        """
        # Load landmarks
        with open(landmarks_file, 'r', encoding='utf-8') as f:
            landmarks_data = json.load(f)
        
        # Load embeddings
        embeddings = np.load(embeddings_file)
        
        # Extract info
        clips = landmarks_data['clips']
        labels = [c['label'] for c in clips]
        unique_labels = sorted(set(labels))
        
        # Calculate statistics
        frame_counts = [c['num_frames'] for c in clips]
        durations = [c.get('duration', 0) for c in clips]
        
        dataset_info = {
            'creation_date': datetime.now().isoformat(),
            'total_clips': len(clips),
            'num_classes': len(unique_labels),
            'classes': unique_labels,
            'features': {
                'mediapipe_landmarks': {
                    'dimension': 468,
                    'description': 'Pose(132) + Hands(126) + Face(210)',
                    'per_frame': True
                },
                'resnext_embeddings': {
                    'dimension': 512,
                    'description': 'R3D-18 spatial-temporal features',
                    'per_clip': True
                }
            },
            'statistics': {
                'frames': {
                    'min': int(np.min(frame_counts)),
                    'max': int(np.max(frame_counts)),
                    'mean': float(np.mean(frame_counts)),
                    'median': float(np.median(frame_counts))
                },
                'duration_seconds': {
                    'min': float(np.min(durations)),
                    'max': float(np.max(durations)),
                    'mean': float(np.mean(durations)),
                    'median': float(np.median(durations))
                }
            },
            'class_distribution': {label: labels.count(label) for label in unique_labels},
            'files': {
                'landmarks': str(landmarks_file.name),
                'embeddings': str(embeddings_file.name),
                'class_index': 'class_index.json'
            }
        }
        
        # Save
        output_file = self.output_dir / "dataset_info.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"💾 Saved dataset info: {output_file}")
        
        return dataset_info
    
    def validate_dataset(self) -> bool:
        """
        Validate dataset integrity
        
        Returns:
            True if valid, False otherwise
        """
        logger.info("🔍 Validating dataset...")
        
        landmarks_file = self.output_dir / "gsl_landmarks.json"
        embeddings_file = self.output_dir / "resnext_embeddings.npy"
        class_index_file = self.output_dir / "class_index.json"
        
        # Check files exist
        if not landmarks_file.exists():
            logger.error(f"❌ Missing: {landmarks_file}")
            return False
        
        if not embeddings_file.exists():
            logger.error(f"❌ Missing: {embeddings_file}")
            return False
        
        # Load data
        with open(landmarks_file, 'r', encoding='utf-8') as f:
            landmarks_data = json.load(f)
        
        embeddings = np.load(embeddings_file)
        
        # Validate counts match
        num_landmark_clips = len(landmarks_data['clips'])
        num_embedding_clips = len(embeddings)
        
        if num_landmark_clips != num_embedding_clips:
            logger.error(f"❌ Clip count mismatch: {num_landmark_clips} landmarks vs {num_embedding_clips} embeddings")
            return False
        
        logger.info(f"✅ Clip counts match: {num_landmark_clips}")
        
        # Validate dimensions
        for idx, clip in enumerate(landmarks_data['clips']):
            landmarks = np.array(clip['landmarks'])
            if landmarks.shape[1] != 468:
                logger.error(f"❌ Invalid landmark dimension for {clip['clip_name']}: {landmarks.shape[1]}")
                return False
        
        if embeddings.shape[1] != 512:
            logger.error(f"❌ Invalid embedding dimension: {embeddings.shape[1]}")
            return False
        
        logger.info(f"✅ Feature dimensions valid: 468 landmarks, 512 embeddings")
        
        # Validate no NaN/Inf
        for idx, clip in enumerate(landmarks_data['clips']):
            landmarks = np.array(clip['landmarks'])
            if np.isnan(landmarks).any() or np.isinf(landmarks).any():
                logger.error(f"❌ Invalid values in {clip['clip_name']}")
                return False
        
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            logger.error(f"❌ Invalid values in embeddings")
            return False
        
        logger.info(f"✅ No NaN/Inf values found")
        
        # Validate labels
        labels = [c['label'] for c in landmarks_data['clips']]
        if not labels:
            logger.error("❌ No labels found")
            return False
        
        logger.info(f"✅ All validations passed!")
        
        return True
    
    def package_dataset(self):
        """Complete dataset packaging workflow"""
        logger.info("=" * 80)
        logger.info("📦 STAGE 5: Dataset Packaging")
        logger.info("=" * 80)
        
        landmarks_file = self.output_dir / "gsl_landmarks.json"
        embeddings_file = self.output_dir / "resnext_embeddings.npy"
        
        # Load landmarks to get labels
        with open(landmarks_file, 'r', encoding='utf-8') as f:
            landmarks_data = json.load(f)
        
        labels = [c['label'] for c in landmarks_data['clips']]
        
        # Create and save class index
        class_index = self.create_class_index(labels)
        self.save_class_index(class_index)
        
        # Create and save dataset info
        dataset_info = self.create_dataset_info(landmarks_file, embeddings_file)
        
        # Validate dataset
        is_valid = self.validate_dataset()
        
        if is_valid:
            logger.info("\n" + "=" * 80)
            logger.info("✅ Dataset Packaging Complete!")
            logger.info("=" * 80)
            logger.info(f"📊 Total clips: {dataset_info['total_clips']}")
            logger.info(f"🏷️  Classes: {dataset_info['num_classes']}")
            logger.info(f"📁 Output: {self.output_dir}/")
            logger.info("=" * 80)
        else:
            logger.error("❌ Dataset validation failed")
            raise ValueError("Dataset validation failed")
        
        return dataset_info
