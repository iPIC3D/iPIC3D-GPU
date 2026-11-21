#include "sparseIndex.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace sparseBins {

std::vector<std::uint8_t> encodeBinIndex(const std::vector<int>& offsets,
                                         const int* index, int regionBins) {
  if (regionBins <= 0)
    throw std::invalid_argument("encodeBinIndex: regions of " +
                                std::to_string(regionBins) + " bins");
  std::vector<std::uint8_t> out;
  for (size_t r = 0; r + 1 < offsets.size(); r++) {
    const int* bins = index + offsets[r];
    const int count = offsets[r + 1] - offsets[r];
    const size_t at = out.size();
    if (isBitmap(count, regionBins)) {
      out.resize(at + bitmapBytes(regionBins), 0);
      for (int i = 0; i < count; i++)
        out[at + bins[i] / kBinsPerBitmapByte] |=
            (std::uint8_t)(1u << (bins[i] % kBinsPerBitmapByte));
      continue;
    }
    auto list = [&](auto type) {
      using T = decltype(type);
      out.resize(at + sizeof(T) * count);
      for (int i = 0; i < count; i++) {
        const T bin = (T)bins[i];
        std::memcpy(&out[at + sizeof(T) * i], &bin, sizeof(T));
      }
    };
    if (binIndexBytes(regionBins) == 2)
      list(std::uint16_t{});
    else
      list(std::uint32_t{});
  }
  return out;
}

} // namespace sparseBins
