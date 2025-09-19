import torch
import torch.nn as nn

class DiversityModel(nn.Module):
    def __init__(self, models, weights):
        super(DiversityModel, self).__init__()
        if len(models) != len(weights) or len(models) < 1:
            raise ValueError("Number of models must match number of weights and be at least 1")
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)
        self.num_models = len(models)
        # Ensure output dimensions match (assuming all models output logits for num_classes)
        self.output_dim = models[0].output_dim if hasattr(models[0], 'output_dim') else 39  # Adjust based on your setup

    def forward(self, x):
        # Get predictions from all models
        outputs = [model(x) for model in self.models]
        # Weighted average of logits
        weighted_outputs = torch.stack([w * output for w, output in zip(self.weights, outputs)])
        return torch.sum(weighted_outputs, dim=0)

    def get_output_dim(self):
        return self.output_dim