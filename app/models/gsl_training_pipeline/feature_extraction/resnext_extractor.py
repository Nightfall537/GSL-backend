"""
ResNeXt-101 3D CNN Feature Extractor (CRITICAL COMPONENT)
Extracts spatial-temporal features for robust GSL recognition
Captures: motion patterns, hand-face interaction, rotation, occlusion handling
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json

from ..utils.logger import setup_logger
from ..config.training_config import FEATURE_CONFIG, SEGMENTED_CLIPS_DIR, METADATA_FILE, OUTPUT_DIR

logger = setup_logger(__name__)


class ResNeXt3DFeatureExtractor:
    """
    Extract spatial-temporal features using ResNeXt-101 3D CNN
    Pre-trained on Kinetics-400 for action recognition
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize ResNeXt 3D CNN model
        
        Args:
            config: Configuration dict (uses FEATURE_CONFIG if None)
        """
        self.config = config or FEATURE_CONFIG['resnext']
        self.model = None
        self.device = None
        
        # Initialize PyTorch model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize PyTorch 3D CNN model"""
        try:
            import torch
            import torchvision.models.video as video_models
            
            logger.info("🔄 Initializing ResNeXt-101 3D CNN...")
            
            # Check for GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🔧 Device: {self.device}")
            
            # Load pre-trained 3D CNN model
            model_name = self.config['model_name']
            
            if model_name == 'r3d_18':
                from torchvision.models.video import r3d_18, R3D_18_Weights
                logger.info("📥 Loading R3D-18 (3D ResNet)...")
                weights = R3D_18_Weights.KINETICS400_V1
                self.model = r3d_18(weights=weights)
            elif model_name == 'mc3_18':
                from torchvision.models.video import mc3_18, MC3_18_Weights
                logger.info("📥 Loading MC3-18 (Mixed Convolution 3D)...")
                weights = MC3_18_Weights.KINETICS400_V1
                self.model = mc3_18(weights=weights)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Remove classification head to get embeddings
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ {model_name.upper()} initialized (Kinetics-400 pre-trained)")
            logger.info("🎯 Captures: spatial-temporal features, motion, hand-face interaction")
            logger.info(f"📊 Embedding dimension: {self.config['embedding_dim']}")
            
        except ImportError as e:
            logger.error("❌ PyTorch not available")
            logger.info("💡 Install with: pip install torch torchvision")
            logger.info("💡 Will skip ResNeXt feature extraction")
            self.model = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize ResNeXt: {e}")
            self.model = None
    
    def preprocess_video(self, video_path: Path) -> Optional['torch.Tensor']:
        """
        Preprocess video for 3D CNN input
        
        Args:
            video_path: Path to video file
        
        Returns:
            torch.Tensor: Shape [1, 3, 16, 112, 112] or None if failed
        """
        if self.model is None:
            return None
        
        try:
            import torch
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Cannot open video: {video_path.name}")
                return None
            
            # Read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize to 112x112
                frame = cv2.resize(frame, tuple(self.config['frame_size']))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                logger.warning(f"No frames read from {video_path.name}")
                return None
            
            # Sample 16 frames uniformly
            frames = self._sample_frames(frames, self.config['num_frames'])
            
            # Convert to tensor [T, H, W, C]
            video_tensor = torch.FloatTensor(np.array(frames))
            
            # Permute to [C, T, H, W]
            video_tensor = video_tensor.permute(3, 0, 1, 2)
            
            # Normalize to [0, 1]
            video_tensor = video_tensor / 255.0
            
            # Normalize with ImageNet stats
            mean = torch.FloatTensor(self.config['normalize_mean']).view(3, 1, 1, 1)
            std = torch.FloatTensor(self.config['normalize_std']).view(3, 1, 1, 1)
            video_tensor = (video_tensor - mean) / std
            
            # Add batch dimension [1, C, T, H, W]
            video_tensor = video_tensor.unsqueeze(0)
            
            return video_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing {video_path.name}: {e}")
            return None
    
    def _sample_frames(self, frames: List[np.ndarray], target_frames: int) -> List[np.ndarray]:
        """
        Uniformly sample target_frames from video
        
        Args:
            frames: List of video frames
            target_frames: Number of frames to sample
        
        Returns:
            List of sampled frames
        """
        n_frames = len(frames)
        
        if n_frames < target_frames:
            # Pad with last frame
            return frames + [frames[-1]] * (target_frames - n_frames)
        else:
            # Uniform sampling
            indices = np.linspace(0, n_frames - 1, target_frames, dtype=int)
            return [frames[i] for i in indices]
    
    def extract_features(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Extract 512-dimensional embedding from video
        
        Args:
            video_path: Path to video file
        
        Returns:
            np.ndarray: Shape [512] or None if extraction fails
        """
        if self.model is None:
            return None
        
        try:
            import torch
            
            # Preprocess video
            video_tensor = self.preprocess_video(video_path)
            if video_tensor is None:
                return None
            
            # Move to device
            video_tensor = video_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(video_tensor)
                features = features.squeeze().cpu().numpy()
            
            # Validate shape
            if features.shape != (self.config['embedding_dim'],):
                logger.warning(f"Unexpected feature shape: {features.shape}")
                return None
            
            return features
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM - falling back to CPU")
            self.device = torch.device("cpu")
            self.model = self.model.cpu()
            return self.extract_features(video_path)
        except Exception as e:
            logger.error(f"Error extracting features from {video_path.name}: {e}")
            return None
    
    def extract_all_features(self, labeled_clips: List[Dict] = None) -> tuple:
        """
        Extract ResNeXt features from all labeled clips
        
        Args:
            labeled_clips: List of clip metadata (loads from file if None)
        
        Returns:
            tuple: (all_embeddings, clip_names)
        """
        if self.model is None:
            logger.warning("⚠️  ResNeXt model not available - skipping extraction")
            return [], []
        
        # Load clips if not provided
        if labeled_clips is None:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            labeled_clips = [c for c in metadata['clips'] if c.get('label')]
        
        all_embeddings = []
        clip_names = []
        
        logger.info("🔄 Extracting ResNeXt 3D CNN features from all clips...")
        logger.info("🎯 This captures spatial-temporal patterns, motion, and interactions")
        
        for idx, clip in enumerate(labeled_clips, 1):
            clip_path = SEGMENTED_CLIPS_DIR / clip['clip_name']
            
            if not clip_path.exists():
                logger.warning(f"⚠️  Clip not found: {clip['clip_name']}")
                # Add zero embedding for missing clips
                all_embeddings.append(np.zeros(self.config['embedding_dim']))
                clip_names.append(clip['clip_name'])
                continue
            
            logger.info(f"🎬 [{idx}/{len(labeled_clips)}] {clip['clip_name']}")
            
            # Extract features
            embedding = self.extract_features(clip_path)
            
            if embedding is not None:
                all_embeddings.append(embedding)
                logger.info(f"   ✅ Extracted {len(embedding)}-dim embedding")
            else:
                # Use zero embedding as fallback
                all_embeddings.append(np.zeros(self.config['embedding_dim']))
                logger.warning(f"   ⚠️  Using zero embedding (extraction failed)")
            
            clip_names.append(clip['clip_name'])
        
        valid_count = sum(1 for emb in all_embeddings if not np.allclose(emb, 0))
        logger.info(f"\n✅ Successfully extracted ResNeXt features: {valid_count}/{len(labeled_clips)} clips")
        
        return all_embeddings, clip_names
    
    def save_to_npy(self, embeddings: List[np.ndarray]):
        """
        Save embeddings to NumPy file
        
        Args:
            embeddings: List of embedding arrays
        """
        output_file = OUTPUT_DIR / "resnext_embeddings.npy"
        
        # Convert to array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Save
        np.save(output_file, embeddings_array)
        
        logger.info(f"💾 Saved ResNeXt embeddings to: {output_file}")
        logger.info(f"   Shape: {embeddings_array.shape}")
        logger.info(f"   Valid embeddings: {np.sum(~np.all(embeddings_array == 0, axis=1))}")
