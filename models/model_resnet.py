import torch
import torch.nn as nn
import torchvision.models as models
from models.model_caw import CAWLayer  # Import CAW Layer

class ResNetWithCAW(nn.Module):
    def __init__(self, base_model="resnet18", num_classes=2, use_caw_layers=True):
        """
        ResNet model with CAW (Concept-Attention Whitening) layers.

        Args:
            base_model (str): 'resnet18' or 'resnet50'.
            num_classes (int): Number of output classes.
            use_caw_layers (bool): Whether to replace BN layers with CAW layers.
        """
        super(ResNetWithCAW, self).__init__()

        # Load ImageNet-pretrained ResNet backbone
        if base_model == "resnet18":
            resnet = models.resnet18(weights="IMAGENET1K_V1")
            caw_layer_indices = [8]  # Replace BN in these layers
        elif base_model == "resnet50":
            resnet = models.resnet50(weights="IMAGENET1K_V1")
            caw_layer_indices = [16]  # Adjust for deeper architecture
        else:
            raise ValueError("Unsupported model. Choose 'resnet18' or 'resnet50'.")

        # Extract feature backbone (remove final FC layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Retain convolutional backbone
        self.avgpool = resnet.avgpool  # Global Average Pooling
        self.dropout = nn.Dropout(p=0.3)  # ✅ Dropout for regularization
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)  # Final classification layer
        self.caw_layers = []  # Store applied CAW layers

        # Replace BN layers with CAW layers (if enabled)
        if use_caw_layers:
            self.replace_bn_with_caw(self.features, caw_layer_indices)

        print(f"✅ Model initialized with {len(self.caw_layers)} CAW layers.")  # Debugging

    def replace_bn_with_caw(self, module, layer_indices):
        """
        Replaces BatchNorm layers with CAW layers at specified indices.
        """
        layer_counter = [0]  # Track BN layers globally

        def recursive_replace(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    if layer_counter[0] in layer_indices:
                        print(f"🔄 Replacing BN layer {layer_counter[0]} with CAW")
                        num_features = child.num_features
                        caw_layer = CAWLayer(num_features, whitening_eps=1e-5, gamma=0.5)
                        setattr(module, name, caw_layer)
                        self.caw_layers.append(caw_layer)  # ✅ Store CAW layers
                    layer_counter[0] += 1  # ✅ Increment correctly
                recursive_replace(child)

        recursive_replace(module)

    def forward(self, x, concept_images=None):
        """
        Forward pass of the model.
        - Supports concept alignment in CAW layers.
        - Returns both classification output and aggregated concept masks.
        """

        concept_masks = []  # ✅ Store concept masks from multiple CAW layers

        # Forward pass through feature backbone
        for name, module in self.features.named_children():
            if isinstance(module, CAWLayer):
                x, concept_mask = module(x, concept_images)  # ✅ Get concept masks from CAW layers
                if concept_mask is not None:
                    concept_masks.append(concept_mask)  # ✅ Collect all masks
            else:
                x = module(x)

        # Global Average Pooling
        x = self.avgpool(x)

        # Flatten and pass through FC layer
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # ✅ Apply dropout
        x = self.fc(x)

        # ✅ Aggregate concept masks (if multiple CAW layers exist)
        aggregated_mask = torch.stack(concept_masks).mean(dim=0) if concept_masks else None

        return x, aggregated_mask  # ✅ Ensure concept masks are always returned
