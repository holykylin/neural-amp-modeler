// 添加立体声处理接口
class StereoModel {
public:
    virtual std::array<float, 2> processStereo(float left, float right) = 0;
    virtual ~StereoModel() = default;
};

// 修改模型加载逻辑
std::unique_ptr<StereoModel> LoadStereoModel(const std::string& modelPath) {
    // 加载支持双通道的模型
    // ...
} 