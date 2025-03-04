class EffectChainAnalyzer(nn.Module):
    def __init__(self, num_effects=8):
        super().__init__()
        # 多尺度特征提取
        self.temporal_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, dilation=2**i),
                nn.BatchNorm1d(32),
                nn.LeakyReLU()
            ) for i in range(4)
        ])
        
        # 效果参数估计
        self.effect_params = nn.Linear(128, num_effects)
        
    def forward(self, x):
        # 融合多尺度时序特征
        features = [block(x) for block in self.temporal_blocks]
        aggregated = torch.mean(torch.stack(features), dim=0)
        return self.effect_params(aggregated) 