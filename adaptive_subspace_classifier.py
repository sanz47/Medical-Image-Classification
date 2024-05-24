import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSubspaceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, subspace_dim=100):
        super().__init__()
        self.subspace_dim = subspace_dim
        self.num_classes = num_classes
        
        self.subspace_bases = nn.Parameter(torch.randn(num_classes, subspace_dim, input_dim))
        self.fc = nn.Linear(input_dim, subspace_dim)

    def forward(self, x):
        x_proj = self.fc(x)
        x_proj = F.normalize(x_proj, dim=-1)

        similarities = torch.einsum('bd,cad->bca', x_proj, self.subspace_bases)
        distances = 1 - similarities
        
        scores = -distances.mean(dim=-1)
        return scores

if __name__ == "__main__":
    input_features = torch.randn(5, 512)  
    asc = AdaptiveSubspaceClassifier(input_dim=512, num_classes=10)  
    preds = asc(input_features)
    print(preds.shape)  
