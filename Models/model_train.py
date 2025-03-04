# 修改模型输出为双通道
class StereoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv1d(2, 32, 3)  # 立体声输入
        self.decoder = nn.Conv1d(32, 2, 3)  # 立体声输出

    def forward(self, x):
        # x shape: [batch, 2, samples]
        return self.decoder(self.encoder(x)) 