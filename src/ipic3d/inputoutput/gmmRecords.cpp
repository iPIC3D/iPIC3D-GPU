#include "gmmRecords.h"

#include <stdexcept>

#include "DataAnalysisInput.h"

namespace dataAnalysis {

static_assert(kGMMMaxComponents <= kGMMMaxSlots &&
                  kGMMMaxComponents <=
                      std::numeric_limits<GMMComponentCount>::max(),
              "a record indexes the components of a fit");

std::string describeGMMRecords(const std::string& members) {
  const std::string dim = std::to_string(kGMMRecordDim);
  return describeRecords(
      members + "  \"covariance_elements\": [\"uu\", \"uv\", \"uw\", \"vv\", "
                "\"vw\", \"ww\"],\n",
      kGMMRecordMagic,
      {{"magic", dtypeName<decltype(GMMRecordHeader::magic)>(), ""},
       {"cycle", dtypeName<decltype(GMMRecordHeader::cycle)>(), ""},
       {"species", dtypeName<decltype(GMMRecordHeader::species)>(), ""},
       {"regions", dtypeName<decltype(GMMRecordHeader::regions)>(), ""},
       {"fits", dtypeName<decltype(GMMRecordHeader::fits)>(), ""},
       {"components", dtypeName<decltype(GMMRecordHeader::components)>(), ""},
       {"record_bytes", dtypeName<decltype(GMMRecordHeader::bytes)>(), ""}},
      {{"log_likelihood", dtypeName<double>(), "\"fits\""},
       {"weights", dtypeName<GMMRecordValue>(), "\"components\""},
       {"means", dtypeName<GMMRecordValue>(), "\"components\", " + dim},
       {"covariances", dtypeName<GMMRecordValue>(),
        "\"components\", " + std::to_string(kGMMPackedCovariance)},
       {"steps", dtypeName<std::int32_t>(), "\"fits\""},
       {"region", dtypeName<GMMRegionIndex>(), "\"fits\""},
       {"counts", dtypeName<GMMComponentCount>(), "\"fits\""},
       {"slot", dtypeName<GMMSlot>(), "\"components\""}},
      "one record per analysis cycle and species of VelocityHistGMMSpecies "
      "with a fit at that cycle: its fitted regions (the rank, or macrocell "
      "m = (mz*My + my)*Mx + mx), increasing, and per fit its log-likelihood "
      "(that of the parameters before the last EM step), its EM steps and "
      "its active components (counts, those not pruned); then those "
      "components fit after fit, by increasing slot (their index among the "
      "GMMComponents of the fit, kept from one analysis to the next by the "
      "warm start): weights, means (u, v, w) and covariances "
      "(covariance_elements, the upper triangle) in the units of the "
      "velocities");
}

void writeGMMRecord(RecordFile& file, int cycle, int species, int regions,
                    int components, const std::vector<GMMFitView>& fits) {
  auto refuse = [&](const std::string& why) {
    throw std::invalid_argument("writeGMMRecord: " + why);
  };
  if (fits.empty() || regions > kGMMMaxRegions || components < 1 ||
      components > kGMMMaxComponents)
    refuse(std::to_string(fits.size()) + " fits of " + std::to_string(regions) +
           " regions, " + std::to_string(components) + " components each");
  std::vector<double> logLikelihood;
  std::vector<std::int32_t> steps;
  std::vector<GMMRegionIndex> region;
  std::vector<GMMComponentCount> counts;
  std::vector<GMMSlot> slot;
  std::vector<GMMRecordValue> weight, mean, coVariance;
  for (const GMMFitView& fit : fits) {
    // the reader finds a fit by bisection on the regions
    if (fit.region < 0 || fit.region >= regions ||
        (!region.empty() && fit.region <= region.back()))
      refuse("region " + std::to_string(fit.region) +
             " out of range or not increasing");
    logLikelihood.push_back(fit.logLikelihood);
    steps.push_back(fit.steps);
    region.push_back((GMMRegionIndex)fit.region);
    const size_t first = slot.size();
    for (int k = 0; k < components; k++) {
      if (!fit.active[k])
        continue;
      slot.push_back((GMMSlot)k);
      weight.push_back((GMMRecordValue)fit.weight[k]);
      const double* m = fit.mean + k * kGMMRecordDim;
      mean.insert(mean.end(), m, m + kGMMRecordDim);
      const double* c = fit.coVariance + k * kGMMRecordDim * kGMMRecordDim;
      for (int i = 0; i < kGMMRecordDim; i++)
        for (int j = i; j < kGMMRecordDim; j++)
          coVariance.push_back((GMMRecordValue)c[i * kGMMRecordDim + j]);
    }
    counts.push_back((GMMComponentCount)(slot.size() - first));
  }
  file.begin(GMMRecordHeader{0, cycle, species, (std::uint32_t)regions,
                             (std::uint32_t)fits.size(),
                             (std::uint32_t)slot.size(), 0});
  file.put(logLikelihood.data(), logLikelihood.size());
  file.put(weight.data(), weight.size());
  file.put(mean.data(), mean.size());
  file.put(coVariance.data(), coVariance.size());
  file.put(steps.data(), steps.size());
  file.put(region.data(), region.size());
  file.put(counts.data(), counts.size());
  file.put(slot.data(), slot.size());
  file.end();
}

} // namespace dataAnalysis
