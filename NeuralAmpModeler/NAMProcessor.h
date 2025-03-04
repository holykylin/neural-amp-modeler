// 原单声道处理
void ProcessBlock(sample** inputs, sample** outputs, int nFrames) {
    for (int i = 0; i < nFrames; ++i) {
        float input = inputs[0][i]; // 单声道输入
        // ... 处理逻辑
        outputs[0][i] = processedSample; // 单声道输出
    }
}

// 修改为立体声处理
void ProcessBlock(sample** inputs, sample** outputs, int nFrames) {
    for (int i = 0; i < nFrames; ++i) {
        // 立体声输入处理
        float inputL = inputs[0][i];
        float inputR = inputs[1][i];
        
        // 双通道模型推理
        std::array<float, 2> processed = model.processStereo(inputL, inputR);
        
        outputs[0][i] = processed[0]; // 左声道输出
        outputs[1][i] = processed[1]; // 右声道输出
    }
} 