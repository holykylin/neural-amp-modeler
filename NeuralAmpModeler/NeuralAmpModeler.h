#include "AudioDSPTools/dsp/wav.h"
#include "AudioDSPTools/dsp/ResamplingContainer/ResamplingContainer.h"

#include "Colors.h"
#include "ToneStack.h"

#include "IPlug_include_in_plug_hdr.h"
#include "ISender.h"


const int kNumPresets = 1;
// 修改为立体声处理
constexpr size_t kNumChannelsInternal = 2;

class NAMSender : public iplug::IPeakAvgSender<>
{
public:
  NAMSender()
  : iplug::IPeakAvgSender<>(-90.0, true, 5.0f, 1.0f, 300.0f, 500.0f)
  {
  }
}; 