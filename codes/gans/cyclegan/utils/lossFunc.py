import torch

def real_discriminator_loss(D_out: torch.Tensor, lambda_weight: float = 1.0) -> torch.Tensor:
    """Loss for classifying real samples as real."""
    return lambda_weight * torch.mean((D_out - 1) ** 2)

def fake_discriminator_loss(D_out: torch.Tensor, lambda_weight: float = 1.0) -> torch.Tensor:
    """Loss for classifying fake samples as fake."""
    return lambda_weight * torch.mean(D_out ** 2)

def cycle_consistency_loss(real_im: torch.Tensor, reconstructed_im: torch.Tensor, lambda_weight: float = 1.0) -> torch.Tensor:
    """Cycle consistency loss: L1 difference between real and reconstructed images."""
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    return lambda_weight * reconstr_loss
