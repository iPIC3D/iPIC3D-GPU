#ifndef HEAT_FLUX_COMPONENTS_H
#define HEAT_FLUX_COMPONENTS_H

#include <array>

namespace HeatFlux {

constexpr int ComponentCount = 10;

inline constexpr std::array<const char*, ComponentCount> ComponentNames = {
    "Qxxx", "Qxxy", "Qxxz", "Qxyy", "Qxyz",
    "Qxzz", "Qyyy", "Qyyz", "Qyzz", "Qzzz"};

inline int componentIndex(int species, int component) {
  return species * ComponentCount + component;
}

} // namespace HeatFlux

#endif // HEAT_FLUX_COMPONENTS_H
