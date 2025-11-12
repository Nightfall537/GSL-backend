"""
Unit Tests for Computer Vision Module

Tests gesture recognition AI model integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.ai.computer_vision import ComputerVisionModel


class TestComputerVisionModel:
    """Test cases for ComputerVisionModel."""
    
    @pytest.fixture
    def cv_model(self):
        """Create ComputerVisionModel instance."""
        return ComputerVisionModel()
    
    def test_initialization(self, cv_model):
        """Test model initialization."""
        assert cv_model.input_shape == (224, 224, 3)
        assert isinstance(cv_model.labels, list)
        assert len(cv_model.labels) > 0
    
    @pytest.mark.asyncio
    async def test_predict_returns_predictions(self, cv_model):
        """Test that predict returns list of predictions."""
        frames = np.zeros((10, 224, 224, 3))
        
        predictions = await cv_model.predict(frames)
        
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        assert all('label' in pred for pred in predictions)
        assert all('confidence' in pred for pred in predictions)
    
    @pytest.mark.asyncio
    async def test_predict_confidence_scores(self, cv_model):
        """Test that predictions have valid confidence scores."""
        frames = np.zeros((10, 224, 224, 3))
        
        predictions = await cv_model.predict(frames)
        
        for pred in predictions:
            assert 0.0 <= pred['confidence'] <= 1.0
    
    def test_preprocess_frames(self, cv_model):
        """Test frame preprocessing."""
        frames = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)
        
        processed = cv_model._preprocess_frames(frames)
        
        assert processed.shape[0] == 5
        assert processed.shape[1:] == cv_model.input_shape
        assert processed.dtype == np.float32
        assert np.all(processed >= 0.0) and np.all(processed <= 1.0)
    
    def test_extract_features(self, cv_model):
        """Test feature extraction."""
        frames = np.zeros((10, 224, 224, 3))
        
        features = cv_model.extract_features(frames)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == frames.shape[0]
    
    def test_calculate_similarity(self, cv_model):
        """Test similarity calculation between feature vectors."""
        features1 = np.random.rand(512)
        features2 = np.random.rand(512)
        
        similarity = cv_model.calculate_similarity(features1, features2)
        
        assert 0.0 <= similarity <= 1.0
    
    def test_calculate_similarity_identical(self, cv_model):
        """Test similarity of identical vectors."""
        features = np.random.rand(512)
        
        similarity = cv_model.calculate_similarity(features, features)
        
        assert similarity == pytest.approx(1.0, abs=0.01)
    
    def test_calculate_similarity_zero_vectors(self, cv_model):
        """Test similarity with zero vectors."""
        features1 = np.zeros(512)
        features2 = np.random.rand(512)
        
        similarity = cv_model.calculate_similarity(features1, features2)
        
        assert similarity == 0.0