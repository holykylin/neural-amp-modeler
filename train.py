def compute_loss(output, target):
    # 时域损失
    time_loss = F.mse_loss(output, target)
    
    # 频域损失（STFT差异）
    output_stft = torch.stft(output, n_fft=2048, return_complex=True)
    target_stft = torch.stft(target, n_fft=2048, return_complex=True)
    freq_loss = F.l1_loss(output_stft.abs(), target_stft.abs())
    
    return 0.7*time_loss + 0.3*freq_loss 