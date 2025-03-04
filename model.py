class GuitarToneModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改前
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),  # 输入通道为1
        
        # 修改后
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5),  # 输入通道改为2（立体声）
            # 后续层保持通道数加倍... 