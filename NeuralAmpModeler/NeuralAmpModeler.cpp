#include <algorithm> // std::clamp, std::min
#include <cmath> // pow
#include <filesystem>
#include <iostream>
#include <utility>

#include "Colors.h"
#include "NeuralAmpModelerCore/NAM/activations.h"
#include "NeuralAmpModelerCore/NAM/get_dsp.h"
// clang-format off
// These includes need to happen in this order or else the latter won't know
// a bunch of stuff.
#include "NeuralAmpModeler.h"
#include "IPlug_include_in_plug_src.h"
// clang-format on
#include "architecture.hpp"

#include "NeuralAmpModelerControls.h"

using namespace iplug;
using namespace igraphics;

const double kDCBlockerFrequency = 5.0;

// 修改输入输出通道配置
// 原配置（单声道）
MakeDefaultInput(ERoute::kInput, 0, 1, "AudioInput"); // 单声道输入
MakeDefaultOutput(ERoute::kOutput, 0, 1, "AudioOutput"); // 单声道输出

// 修改为立体声配置
MakeDefaultInput(ERoute::kInput, 0, 2, "AudioInput"); // 立体声输入
MakeDefaultOutput(ERoute::kOutput, 0, 2, "AudioOutput"); // 立体声输出 

void NeuralAmpModeler::_ProcessInput(iplug::sample** inputs, const size_t nFrames, const size_t nChansIn,
                                     const size_t nChansOut)
{
  // 支持立体声处理
  if (nChansOut != 2)
  {
    std::stringstream ss;
    ss << "Expected stereo output, but " << nChansOut << " output channels are requested!";
    throw std::runtime_error(ss.str());
  }

  double gain = mInputGain;
  // 保留立体声信息，不再将输入混合为单声道
  // 假设_PrepareBuffers()已经被调用
  for (size_t c = 0; c < nChansIn && c < nChansOut; c++)
  {
    for (size_t s = 0; s < nFrames; s++)
    {
      mInputArray[c][s] = gain * inputs[c][s];
    }
  }
  
  // 如果输入是单声道但输出需要立体声，则复制到两个通道
  if (nChansIn == 1 && nChansOut == 2)
  {
    for (size_t s = 0; s < nFrames; s++)
    {
      mInputArray[1][s] = mInputArray[0][s];
    }
  }
}

void NeuralAmpModeler::ProcessBlock(iplug::sample** inputs, iplug::sample** outputs, int nFrames)
{
  const size_t numChannelsExternalIn = (size_t)NInChansConnected();
  const size_t numChannelsExternalOut = (size_t)NOutChansConnected();
  const size_t numChannelsInternal = kNumChannelsInternal;
  const size_t numFrames = (size_t)nFrames;
  const double sampleRate = GetSampleRate();

  // Disable floating point denormals
  std::fenv_t fe_state;
  std::feholdexcept(&fe_state);
  disable_denormals();

  _PrepareBuffers(numChannelsInternal, numFrames);
  // 保留立体声信息
  _ProcessInput(inputs, numFrames, numChannelsExternalIn, numChannelsInternal);
  _ApplyDSPStaging();
  const bool noiseGateActive = GetParam(kNoiseGateActive)->Value();
  const bool toneStackActive = GetParam(kEQActive)->Value();

  // 噪声门处理（对每个通道单独处理）
  sample** triggerOutputL = mInputPointers;
  sample** triggerOutputR = mInputPointers + 1;
  
  if (noiseGateActive)
  {
    const double time = 0.01;
    const double threshold = GetParam(kNoiseGateThreshold)->Value();
    const double ratio = 0.1;
    const double openTime = 0.005;
    const double holdTime = 0.01;
    const double closeTime = 0.05;
    const dsp::noise_gate::TriggerParams triggerParams(time, threshold, ratio, openTime, holdTime, closeTime);
    mNoiseGateTrigger.SetParams(triggerParams);
    mNoiseGateTrigger.SetSampleRate(sampleRate);
    
    // 对左右声道分别处理
    triggerOutputL = mNoiseGateTrigger.Process(mInputPointers, 1, numFrames);
    triggerOutputR = mNoiseGateTrigger.Process(mInputPointers + 1, 1, numFrames);
  }

  if (mModel != nullptr)
  {
    // 处理左右声道
    mModel->process(triggerOutputL[0], mOutputPointers[0], nFrames);
    mModel->process(triggerOutputR[0], mOutputPointers[1], nFrames);
  }
  else
  {
    _FallbackDSP(triggerOutputL, mOutputPointers, 1, numFrames);
    _FallbackDSP(triggerOutputR, mOutputPointers + 1, 1, numFrames);
  }

  // 后续处理...
  // ... existing code ...
  
  // 更新输出
  _ProcessOutput(mOutputPointers, outputs, numFrames, numChannelsInternal, numChannelsExternalOut);
  _UpdateMeters(mInputPointers, mOutputPointers, numFrames, numChannelsInternal, numChannelsInternal);
  
  // 恢复浮点设置
  std::feupdateenv(&fe_state);
}

void NeuralAmpModeler::_ProcessOutput(iplug::sample** inputs, iplug::sample** outputs, const size_t nFrames,
                                      const size_t nChansIn, const size_t nChansOut)
{
  // 支持立体声输出
  const double gain = mOutputGain;
  
  // 处理所有可用的输出通道
  for (size_t c = 0; c < nChansOut && c < nChansIn; c++)
  {
    for (size_t s = 0; s < nFrames; s++)
    {
      outputs[c][s] = gain * inputs[c][s];
    }
  }
  
  // 如果输入是单声道但输出需要多通道，则复制到所有通道
  if (nChansIn == 1 && nChansOut > 1)
  {
    for (size_t c = 1; c < nChansOut; c++)
    {
      for (size_t s = 0; s < nFrames; s++)
      {
        outputs[c][s] = outputs[0][s];
      }
    }
  }
}

void NeuralAmpModeler::_UpdateMeters(sample** inputPointer, sample** outputPointer, const size_t nFrames, 
                                     const size_t nChansIn, const size_t nChansOut)
{
  // 支持立体声电平表
  // 对于立体声，我们可以选择显示两个通道的平均值或最大值
  // 这里我们选择显示最大值
  
  // 创建临时缓冲区来存储合并后的信号
  std::vector<sample> inputMerged(nFrames);
  std::vector<sample> outputMerged(nFrames);
  
  for (size_t s = 0; s < nFrames; s++)
  {
    // 取左右声道的最大值
    inputMerged[s] = std::max(std::abs(inputPointer[0][s]), std::abs(inputPointer[1][s]));
    outputMerged[s] = std::max(std::abs(outputPointer[0][s]), std::abs(outputPointer[1][s]));
  }
  
  // 使用合并后的信号更新电平表
  const int nChansHack = 1; // 仍然使用单通道电平表
  mInputSender.ProcessBlock(&inputMerged[0], (int)nFrames, kCtrlTagInputMeter, nChansHack);
  mOutputSender.ProcessBlock(&outputMerged[0], (int)nFrames, kCtrlTagOutputMeter, nChansHack);
}

NeuralAmpModeler::NeuralAmpModeler(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  // ... existing code ...
  
  // 修改为立体声配置
  MakeDefaultInput(ERoute::kInput, 0, 2, "AudioInput"); // 立体声输入
  MakeDefaultOutput(ERoute::kOutput, 0, 2, "AudioOutput"); // 立体声输出
  
  // ... existing code ...
} 