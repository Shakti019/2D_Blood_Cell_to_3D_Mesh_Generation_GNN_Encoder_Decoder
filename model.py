# NOTE: You will need PyTorch and PyTorch Geometric installed to run this script.
# (pip install torch torch_geometric)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, DataLoader
    # UPGRADE: Importing GATv2Conv which is mathematically more expressive than standard GAT
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, GraphNorm
    print("PyTorch Geometric loaded successfully!")
except ImportError:
    print("PyTorch Geometric is missing. Run: pip install torch torchvision torch-geometric")

# ==============================================================================
# 1. MULTI-TASK GRAPH NEURAL NETWORK ARCHITECTURE
# ==============================================================================

class BloodCellHybridGNN(torch.nn.Module):
    def __init__(self, node_features_dim, hidden_channels, num_classes, edge_features_dim=4):
        super(BloodCellHybridGNN, self).__init__()
        
        # -------------------------------------------------------------
        # THE ENCODER: Multi-Head GATv2 + GraphNorm
        # `GATv2Conv` scales dynamically and prevents attention dilution
        # -------------------------------------------------------------
        self.conv1 = GATv2Conv(node_features_dim, hidden_channels, heads=4, edge_dim=edge_features_dim)
        self.norm1 = GraphNorm(hidden_channels * 4)
        
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=edge_features_dim)
        self.norm2 = GraphNorm(hidden_channels * 4)
        
        self.conv3 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=1, concat=False, edge_dim=edge_features_dim)
        self.norm3 = GraphNorm(hidden_channels)
        
        # -------------------------------------------------------------
        # HEAD 1: PERCENTAGE CLASSIFIER (Deep MLP)
        # Upgraded to a 3-layer deep MLP with GELU activations
        # -------------------------------------------------------------
        self.classifier_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # -------------------------------------------------------------
        # HEAD 2: 3D GRAPH DECODER (Anomaly Detection)
        # -------------------------------------------------------------
        self.decoder_lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.decoder_lin2 = nn.Linear(hidden_channels, node_features_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # --- Encode the Graph ---
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(h, batch)
        h = F.gelu(h) # Using GELU instead of ELU for deeper networks
        
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h, batch)
        h = F.gelu(h)
        
        latent_node_features = self.conv3(h, edge_index, edge_attr=edge_attr)
        latent_node_features = self.norm3(latent_node_features, batch)
        
        # --- Classification Head ---
        x_mean = global_mean_pool(latent_node_features, batch)
        x_max = global_max_pool(latent_node_features, batch)
        graph_signature = torch.cat([x_mean, x_max], dim=1)
        
        # Route through Deep MLP
        raw_logits = self.classifier_mlp(graph_signature)
        classification_percentages = F.softmax(raw_logits, dim=1)
        
        # --- Decoder Head ---
        dec_h = F.gelu(self.decoder_lin1(latent_node_features))
        reconstructed_nodes = self.decoder_lin2(dec_h)
        
        return classification_percentages, reconstructed_nodes

