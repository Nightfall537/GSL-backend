#!/usr/bin/env python3
"""
Complete Feature Extraction Pipeline
Extracts MediaPipe Holistic landmarks (468 features) and ResNeXt-101 3D CNN embeddings (512 features)
from all labeled GSL clips
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from gsl_training_pipeline.feature_extraction.mediapipe_extractor import MediaPipeFeatureExtractor
from gsl_training_pipeline.feature_extraction.resnext_extractor import ResNeXt3DFeatureExtractor
from gsl_training_pipeline.utils.logger import setup_logger

logger = setup_logger(__name__, log_file='logs/feature_extraction.log')


def main():
    """Run complete feature extraction pipeline"""
    logger.info("=" * 80)
    logger.info("🚀 GSL Feature Extraction Pipeline")
    logger.info("=" * 80)
    logger.info("📊 Stage 3: MediaPipe Holistic (468 landmarks per frame)")
    logger.info("🤖 Stage 4: ResNeXt-101 3D CNN (512-dim spatial-temporal embeddings)")
    logger.info("=" * 80)
    
    try:
        # ========================================================================
        # STAGE 3: MediaPipe Holistic Landmark Extraction
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: MediaPipe Holistic Landmark Extraction")
        logger.info("=" * 80)
        
        mediapipe_extractor = MediaPipeFeatureExtractor()
        landmarks, labels, clip_info, unique_labels = mediapipe_extractor.extract_all_features()
        
        if not landmarks:
            logger.error("❌ No landmarks extracted - aborting pipeline")
            return False
        
        # Save landmarks
        mediapipe_extractor.save_to_json(landmarks, labels, clip_info, unique_labels)
        
        logger.info(f"\n✅ Stage 3 Complete!")
        logger.info(f"   Extracted landmarks from {len(landmarks)} clips")
        logger.info(f"   Feature dimension: 468 per frame")
        
        # ========================================================================
        # STAGE 4: ResNeXt-101 3D CNN Feature Extraction (CRITICAL)
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: ResNeXt-101 3D CNN Feature Extraction (CRITICAL)")
        logger.info("=" * 80)
        logger.info("🎯 Captures: spatial-temporal patterns, motion, hand-face interaction")
        logger.info("🎯 Handles: occlusion, rotation, lighting variations")
        
        resnext_extractor = ResNeXt3DFeatureExtractor()
        
        if resnext_extractor.model is not None:
            # Load labeled clips
            import json
            from gsl_training_pipeline.config.training_config import METADATA_FILE
            
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            labeled_clips = [c for c in metadata['clips'] if c.get('label')]
            
            # Extract ResNeXt features
            embeddings, clip_names = resnext_extractor.extract_all_features(labeled_clips)
            
            if embeddings:
                # Save embeddings
                resnext_extractor.save_to_npy(embeddings)
                
                logger.info(f"\n✅ Stage 4 Complete!")
                logger.info(f"   Extracted embeddings from {len(embeddings)} clips")
                logger.info(f"   Embedding dimension: 512")
            else:
                logger.warning("⚠️  No ResNeXt embeddings extracted")
        else:
            logger.warning("⚠️  ResNeXt model not available - skipping Stage 4")
            logger.info("💡 Install PyTorch to enable ResNeXt feature extraction")
            logger.info("💡 Training will proceed with landmarks only")
        
        # ========================================================================
        # Summary
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Feature Extraction Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"✅ MediaPipe landmarks: {len(landmarks)} clips × variable frames × 468 features")
        if resnext_extractor.model is not None and embeddings:
            logger.info(f"✅ ResNeXt embeddings: {len(embeddings)} clips × 512 features")
        logger.info(f"✅ Unique gestures: {len(unique_labels)}")
        logger.info(f"📁 Output directory: gsl_dataset/")
        logger.info("=" * 80)
        logger.info("\n📋 Next Step: Dataset Packaging and Model Training")
        logger.info("Run: python train_gsl_model.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
