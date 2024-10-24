import torch
import logging

logger = logging.getLogger(__name__)


class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean = None
        self.proj = None

    def fit(self, X: torch.Tensor, offset: int = 0) -> None:
        # Center the data
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        covariance_matrix = torch.cov(X_centered.T)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        self.proj = eigenvectors[:, sorted_indices][:, : self.n_components]

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        # Center the data
        X_centered = X - self.mean
        return X_centered @ self.proj

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)  # Return the transformed data
