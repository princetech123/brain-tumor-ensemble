import unittest
import torch
from src.models.fusion import EnsembleFusionModel

class TestEnsembleFusion(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.config = {
            "models": {
                "resnet": {"feature_dim": 2048},
                "efficientnet": {"feature_dim": 1536},
                "vit": {"feature_dim": 768},
                "fusion": {
                    "attention_heads": 4,
                    "dropout": 0.4,
                    "dense_dim_1": 512,
                    "dense_dim_2": 256
                }
            }
        }
        self.fusion_model = EnsembleFusionModel(self.config)
        
        # Batch size of 4 mock feature vectors
        self.dummy_resnet_feats = torch.randn(4, 2048)
        self.dummy_effnet_feats = torch.randn(4, 1536)
        self.dummy_vit_feats = torch.randn(4, 768)

    def test_fusion_forward_shapes(self):
        probs, attn_weights = self.fusion_model(
            self.dummy_resnet_feats, 
            self.dummy_effnet_feats, 
            self.dummy_vit_feats
        )
        
        # Binary prediction output should be shape (Batch, 1)
        self.assertEqual(probs.shape, (4, 1))
        
        # Predictions must be sigmoids (in range [0, 1])
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))
        
        # Attention weights should have shape (Batch, 3, 3) representing cross-model attention
        self.assertEqual(attn_weights.shape, (4, 3, 3))
        
        # Attention weights along key-value dimension should sum to 1 (Softmax normalization check)
        sum_attn = torch.sum(attn_weights, dim=-1)
        # Check close to 1.0 within floating point precision limits
        self.assertTrue(torch.allclose(sum_attn, torch.ones_like(sum_attn), atol=1e-5))

if __name__ == "__main__":
    unittest.main()
