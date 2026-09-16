import torch
import numpy as np
import cv2

class GradCAMEngine:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))
        
    def generate_heatmap(self, input_tensor, class_idx=0):
        self.model.zero_grad()
        score = self.model(input_tensor)[:, class_idx]
        score.backward()
        
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        for i in range(activations.shape[0]): activations[i] *= pooled_grads[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        return heatmap / (np.max(heatmap) + 1e-8) if np.max(heatmap) != 0 else heatmap