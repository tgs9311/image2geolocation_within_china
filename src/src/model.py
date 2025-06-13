from transformers import CLIPVisionModel, CLIPImageProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_hidden_dim: int, dropout_rate: float, activation_fn: nn.Module = nn.GELU):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.act = activation_fn()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x + residual

class ProvinceModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 embed_dim: int = 768,
                 num_blocks: int = 4,
                 mlp_expansion_ratio: float = 4.0,
                 dropout_rate: float = 0.3,
                 input_dropout_rate: float = 0.3,
                 classifier_hidden_dim_ratio: float = 0.5):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout_rate) if input_dropout_rate > 0 else nn.Identity()
        mlp_hidden_dim = int(embed_dim * mlp_expansion_ratio)
        self.blocks = nn.Sequential(*[
            ResidualFeedForwardBlock(embed_dim, mlp_hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        classifier_hidden_dim = int(embed_dim * classifier_hidden_dim_ratio)
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, image_features: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x = self.input_dropout(image_features)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.classifier_head(x)
        return logits if return_logits else F.softmax(logits, dim=-1)

class CityModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 embed_dim: int = 768,
                 num_blocks: int = 6,
                 mlp_expansion_ratio: float = 6.0,
                 dropout_rate: float = 0.3,
                 input_dropout_rate: float = 0.1,
                 classifier_hidden_dim_ratio: float = 0.5):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout_rate) if input_dropout_rate > 0 else nn.Identity()
        mlp_hidden_dim = int(embed_dim * mlp_expansion_ratio)
        self.blocks = nn.Sequential(*[
            ResidualFeedForwardBlock(embed_dim, mlp_hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        classifier_hidden_dim = int(embed_dim * classifier_hidden_dim_ratio)
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, image_features: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x = self.input_dropout(image_features)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.classifier_head(x)
        return logits if return_logits else F.softmax(logits, dim=-1)

class CountryModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 embed_dim: int = 768,
                 num_blocks: int = 8,
                 mlp_expansion_ratio: float = 8.0,
                 dropout_rate: float = 0.3,
                 input_dropout_rate: float = 0.1,
                 classifier_hidden_dim_ratio: float = 0.5):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout_rate) if input_dropout_rate > 0 else nn.Identity()
        mlp_hidden_dim = int(embed_dim * mlp_expansion_ratio)
        self.blocks = nn.Sequential(*[
            ResidualFeedForwardBlock(embed_dim, mlp_hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        classifier_hidden_dim = int(embed_dim * classifier_hidden_dim_ratio)
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, image_features: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x = self.input_dropout(image_features)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.classifier_head(x)
        return logits if return_logits else F.softmax(logits, dim=-1)