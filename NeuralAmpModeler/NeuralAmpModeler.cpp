// 修改输入输出通道配置
// 原配置（单声道）
MakeDefaultInput(ERoute::kInput, 0, 1, "AudioInput"); // 单声道输入
MakeDefaultOutput(ERoute::kOutput, 0, 1, "AudioOutput"); // 单声道输出

// 修改为立体声配置
MakeDefaultInput(ERoute::kInput, 0, 2, "AudioInput"); // 立体声输入
MakeDefaultOutput(ERoute::kOutput, 0, 2, "AudioOutput"); // 立体声输出 