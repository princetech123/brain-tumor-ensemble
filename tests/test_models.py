import unittest
import torch
from src.models.backbones import ResNet50Backbone, EfficientNetB3Backbone, ViTBackbone

class TestModelBackbones(unittest.TestCase):
    def setUp(self):
        # Batch size of 2, 3 channels, 224x224 input
        self.dummy_input = torch.randn(2, 3, 224, 224)

    def test_resnet50_backbone(self):
        # Initialize without loading pretrained weights to speed up testing
        model = ResNet50Backbone(pretrained=False)
        
        # 1. Test standalone forward classification
        logits, features = model(self.dummy_input)
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(features.shape, (2, 2048))
        
        # 2. Test feature extraction mode
        feats_only = model(self.dummy_input, return_features=True)
        self.assertEqual(feats_only.shape, (2, 2048))

    def test_efficientnet_b3_backbone(self):
        model = EfficientNetB3Backbone(pretrained=False)
        
        # 1. Test standalone forward classification
        logits, features = model(self.dummy_input)
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(features.shape, (2, 1536))
        
        # 2. Test feature extraction mode
        feats_only = model(self.dummy_input, return_features=True)
        self.assertEqual(feats_only.shape, (2, 1536))

    def test_vit_backbone(self):
        model = ViTBackbone(pretrained=False)
        
        # 1. Test standalone forward classification
        logits, features = model(self.dummy_input)
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(features.shape, (2, 768))
        
        # 2. Test feature extraction mode
        feats_only = model(self.dummy_input, return_features=True)
        self.assertEqual(feats_only.shape, (2, 768))

if __name__ == "__main__":
    unittest.main()
