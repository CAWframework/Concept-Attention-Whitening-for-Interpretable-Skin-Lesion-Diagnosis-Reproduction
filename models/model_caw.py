import torch
import torch.nn as nn
import torch.nn.functional as F

class CAWLayer(nn.Module):
    def __init__(self, num_features, num_concepts=12, whitening_eps=1e-5, gamma=0.5, eta=0.01):
        super(CAWLayer, self).__init__()
        self.num_features = num_features
        self.num_concepts = num_concepts
        self.whitening_eps = whitening_eps
        self.gamma = gamma
        self.eta = eta

        # Initialize ZCA whitening matrix
        self.whitening_matrix = nn.Parameter(self.init_zca_whitening(num_features), requires_grad=True)

        # Initialize orthogonal matrix
        self.orthogonal_matrix = nn.Parameter(self.initialize_orthogonal(num_features), requires_grad=True)

    def init_zca_whitening(self, num_features):
        """Initializes W as a ZCA Whitening matrix (identity)."""
        return torch.eye(num_features)

    def initialize_orthogonal(self, num_features):
        """Initializes Q as an orthogonal matrix using QR decomposition."""
        Q = torch.randn(num_features, num_features)
        Q, _ = torch.linalg.qr(Q)  # QR decomposition ensures orthogonality
        return Q

    def whitening_transform(self, Z):
        """Apply ZCA Whitening: ψ(Z) = W(Z - μ)."""
        B, C, H, W = Z.shape
        Z = Z.view(B, C, -1)

        mean = Z.mean(dim=2, keepdim=True)
        Z_centered = Z - mean  # Center features

        # Ensure whitening_matrix is tracked in computation graph
        self.whitening_matrix.requires_grad_(True)

        Z_whitened = torch.matmul(self.whitening_matrix.unsqueeze(0), Z_centered)
        return Z_whitened.view(B, C, H, W)

    def orthogonal_transform(self, Z_whitened):
        """Apply orthogonal transformation: Z' = Q^T ψ(Z)."""
        B, C, H, W = Z_whitened.shape
        Z_whitened = Z_whitened.view(B, C, -1)

        A = self.compute_cayley_transform()  # Compute skew-symmetric matrix A
        Q_updated = self.cayley_update(self.orthogonal_matrix, A)

        # ✅ Store `Q_updated` without breaking gradients
        self.orthogonal_matrix = nn.Parameter(Q_updated.detach())

        Z_transformed = torch.matmul(Q_updated.T.unsqueeze(0), Z_whitened)
        return Z_transformed.view(B, C, H, W)

    def compute_cayley_transform(self):
        """
        Compute skew-symmetric matrix A for Cayley transform.
        """

        # ✅ Ensure gradients are tracked (only in training)
        if self.training:  
            self.orthogonal_matrix.requires_grad_(True)
            self.whitening_matrix.requires_grad_(True)

        # ✅ Compute L2 loss directly on the whitening matrix
        loss = torch.norm(self.whitening_matrix, p=2)

        # ✅ Add a small L2 penalty to ensure `self.orthogonal_matrix` is in the computation graph
        loss += 0.001 * torch.norm(self.orthogonal_matrix, p=2)

        # 🔍 Skip gradient computation if in evaluation mode
        if not self.training:
            return torch.eye(self.orthogonal_matrix.shape[0], device=self.orthogonal_matrix.device)

        G = torch.autograd.grad(
            outputs=loss,
            inputs=self.orthogonal_matrix,
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]

        if G is None:
            print("⚠️ Warning: Gradient G is None! Using identity matrix as fallback.")
            return torch.eye(self.orthogonal_matrix.shape[0], device=self.orthogonal_matrix.device)

        # ✅ Ensure matrix A remains skew-symmetric
        A = G.T - self.orthogonal_matrix @ G.T @ self.orthogonal_matrix

        return A


    def cayley_update(self, Q, A):
        """Applies Cayley transform update: Q_{t+1} = (I + η/2 * A)^{-1} (I - η/2 * A) Q."""
        I = torch.eye(Q.shape[0], device=Q.device)
        left = torch.inverse(I + (self.eta / 2) * A)
        right = (I - (self.eta / 2) * A) @ Q
        return left @ right

    def generate_concept_mask(self, concept_features, classifier_weights):
        """Generate concept masks using classifier weights."""
        B, C, H, W = concept_features.shape
        concept_features = concept_features.view(B, C, -1)

        # Ensure classifier weights are used properly
        if classifier_weights.shape[1] != C:
            raise ValueError(f"Mismatch: classifier_weights.shape={classifier_weights.shape}, concept_features.shape={concept_features.shape}")

        M = torch.matmul(classifier_weights.unsqueeze(0), concept_features)

        # Normalize concept activations
        M_min, M_max = M.min(dim=2, keepdim=True)[0], M.max(dim=2, keepdim=True)[0]
        M_range = torch.clamp(M_max - M_min, min=self.whitening_eps)
        M_normalized = (M - M_min) / M_range

        # Binarize concept mask
        M_binary = (M_normalized > self.gamma).float()

        return M_binary.view(B, 1, H, W)  # ✅ Ensure correct shape for concept masks

    def forward(self, Z, concept_features=None, classifier_weights=None):
        """
        Forward pass for CAW layer.
        """
        Z_whitened = self.whitening_transform(Z)
        Z_transformed = self.orthogonal_transform(Z_whitened)

        if concept_features is not None and classifier_weights is not None:
            concept_mask = self.generate_concept_mask(concept_features, classifier_weights)

            # ✅ Debugging Sample Concept Mask
            if self.training and concept_mask is not None:
                print(f"🔍 Sample Concept Mask: {concept_mask[0, 0, :5, :5].detach().cpu().numpy()}")  

            return Z_transformed, concept_mask  # Return both transformed features and concept mask

        return Z_transformed  # Return only transformed features
