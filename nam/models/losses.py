# File: losses.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Loss functions
"""

from typing import Optional as _Optional

import torch as _torch
from auraloss.freq import MultiResolutionSTFTLoss as _MultiResolutionSTFTLoss


def apply_pre_emphasis_filter(x: _torch.Tensor, coef: float) -> _torch.Tensor:
    """
    Apply first-order pre-emphsis filter

    :param x: (*, L)
    :param coef: The coefficient

    :return: (*, L-1)
    """
    return x[..., 1:] - coef * x[..., :-1]


def esr(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    ESR of (a batch of) predictions & targets

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    if preds.ndim == 1 and targets.ndim == 1:
        preds, targets = preds[None], targets[None]
    if preds.ndim != 2:
        raise ValueError(
            f"Expect 2D predictions (batch_size, num_samples). Got {preds.shape}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"Expect 2D targets (batch_size, num_samples). Got {targets.shape}"
        )
    return _torch.mean(
        _torch.mean(_torch.square(preds - targets), dim=1)
        / _torch.mean(_torch.square(targets), dim=1)
    )


def multi_resolution_stft_loss(
    preds: _torch.Tensor,
    targets: _torch.Tensor,
    loss_func: _Optional[_MultiResolutionSTFTLoss] = None,
    device: _Optional[_torch.device] = None,
) -> _torch.Tensor:
    """
    Experimental Multi Resolution Short Time Fourier Transform Loss using auraloss implementation.
    B: Batch size
    L: Sequence length
    C: Number of channels (1 for mono, 2 for stereo)

    :param preds: (B,C,L) for stereo or (B,L) for mono
    :param targets: (B,C,L) for stereo or (B,L) for mono
    :param loss_func: A pre-initialized instance of the loss function module
    :param device: If provided, send the preds and targets to the provided device
    :return: ()
    """
    loss_func = _MultiResolutionSTFTLoss() if loss_func is None else loss_func
    if device is not None:
        preds, targets = [z.to(device) for z in (preds, targets)]
    
    # Handle both mono and stereo cases
    if len(preds.shape) == 2:  # Mono
        return loss_func(preds, targets)
    else:  # Stereo
        # Process each channel separately and average
        loss_l = loss_func(preds[:, 0], targets[:, 0])
        loss_r = loss_func(preds[:, 1], targets[:, 1])
        return (loss_l + loss_r) / 2


def mse_fft(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    Fourier loss

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    fp = _torch.fft.fft(preds)
    ft = _torch.fft.fft(targets)
    e = fp - ft
    return _torch.mean(_torch.square(e.abs()))
