import torch

from models.ppo.trainer import ppo_loss


def test_ppo_loss_stays_finite_with_zero_advantages_and_large_ratio():
    loss_dict = ppo_loss(
        log_p_new=torch.tensor([1000.0, -1000.0]),
        log_p_cur=torch.tensor([0.0, 0.0]),
        A_hat=torch.zeros(2),
        R_hat=torch.zeros(2),
        V_phi=torch.zeros(2),
        H=torch.zeros(2),
    )

    assert torch.isfinite(loss_dict["loss"])
    assert torch.isfinite(loss_dict["L_clip"])
    assert torch.isfinite(loss_dict["rho"])
