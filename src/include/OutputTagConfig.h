#ifndef OUTPUT_TAG_CONFIG_H
#define OUTPUT_TAG_CONFIG_H

#include <set>
#include <string>
#include <sstream>
#include <regex>
#include <cstdio>
#include "HeatFluxComponents.h"

/**
 * @brief Parsed output tag configuration.
 *
 * Replaces the old string-find approach for FieldOutputTag / MomentsOutputTag.
 *
 * Field tags: B, E  (grid-level quantities only)
 *
 * Moments tags use numeric species indexing:
 *   bare name (rho, J, PXX, ..., P, Q)  → all species
 *   name + digit (rho0, J2, PXX3, Qxyz1) → single species
 *   name_tot (rho_tot, J_tot, PXX_tot, P_tot, Qxyz_tot) → sum over species
 */
struct OutputTagConfig {

    // --- Field flags ---
    bool writeB = false;
    bool writeE = false;

    // --- Per-species moment sets (contain species indices to write) ---
    std::set<int> rhoSpecies;
    std::set<int> JSpecies;
    std::set<int> PXXSpecies;
    std::set<int> PXYSpecies;
    std::set<int> PXZSpecies;
    std::set<int> PYYSpecies;
    std::set<int> PYZSpecies;
    std::set<int> PZZSpecies;
    std::array<std::set<int>, HeatFlux::ComponentCount> heatFluxSpecies;

    // --- Total flags ---
    bool writeRhoTot = false;
    bool writeJTot   = false;
    bool writePXXTot = false;
    bool writePXYTot = false;
    bool writePXZTot = false;
    bool writePYYTot = false;
    bool writePYZTot = false;
    bool writePZZTot = false;
    std::array<bool, HeatFlux::ComponentCount> writeHeatFluxTot = {};

    // --- Convenience queries ---

    bool needsAnyField() const {
        return writeB || writeE || !JSpecies.empty() || writeJTot;
    }

    bool needsAnyMoments() const {
        return !rhoSpecies.empty()
            || !PXXSpecies.empty() || !PXYSpecies.empty()
            || !PXZSpecies.empty() || !PYYSpecies.empty()
            || !PYZSpecies.empty() || !PZZSpecies.empty()
            || writeRhoTot || writeJTot
            || writePXXTot || writePXYTot || writePXZTot
            || writePYYTot || writePYZTot || writePZZTot
            || needsAnyHeatFlux();
    }

    bool needsJTotComputation() const {
        return writeJTot;
    }

    bool needsAnyPTot() const {
        return writePXXTot || writePXYTot || writePXZTot
            || writePYYTot || writePYZTot || writePZZTot;
    }

    bool needsAnyHeatFlux() const {
        for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
            if (!heatFluxSpecies[c].empty() || writeHeatFluxTot[c]) return true;
        }
        return false;
    }

    bool needsAnyHeatFluxTot() const {
        for (bool doWrite : writeHeatFluxTot)
            if (doWrite) return true;
        return false;
    }

    /** Number of scalar moment writes (rho + pressure; excludes J which is vector). */
    int countMomentWrites() const {
        int n = 0;
        n += (int)rhoSpecies.size();
        n += (int)PXXSpecies.size();
        n += (int)PXYSpecies.size();
        n += (int)PXZSpecies.size();
        n += (int)PYYSpecies.size();
        n += (int)PYZSpecies.size();
        n += (int)PZZSpecies.size();
        if (writeRhoTot) n++;
        if (writePXXTot) n++;
        if (writePXYTot) n++;
        if (writePXZTot) n++;
        if (writePYYTot) n++;
        if (writePYZTot) n++;
        if (writePZZTot) n++;
        for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
            n += (int)heatFluxSpecies[c].size();
            if (writeHeatFluxTot[c]) n++;
        }
        return n;
    }

    /** Number of vector writes for fields (B, E, per-species J, J_tot). */
    int countFieldWrites() const {
        int n = 0;
        if (writeB) n++;
        if (writeE) n++;
        n += (int)JSpecies.size();
        if (writeJTot) n++;
        return n;
    }
};

// ---------- Helper: add all species 0..ns-1 to a set ----------
inline void addAllSpecies(std::set<int>& s, int ns) {
    for (int i = 0; i < ns; i++) s.insert(i);
}

// ---------- Helper: add all P components for all species ----------
inline void addAllPressureSpecies(OutputTagConfig& cfg, int ns) {
    addAllSpecies(cfg.PXXSpecies, ns);
    addAllSpecies(cfg.PXYSpecies, ns);
    addAllSpecies(cfg.PXZSpecies, ns);
    addAllSpecies(cfg.PYYSpecies, ns);
    addAllSpecies(cfg.PYZSpecies, ns);
    addAllSpecies(cfg.PZZSpecies, ns);
}

// ---------- Helper: add all P components for one species ----------
inline void addPressureForSpecies(OutputTagConfig& cfg, int si) {
    cfg.PXXSpecies.insert(si);
    cfg.PXYSpecies.insert(si);
    cfg.PXZSpecies.insert(si);
    cfg.PYYSpecies.insert(si);
    cfg.PYZSpecies.insert(si);
    cfg.PZZSpecies.insert(si);
}

// ---------- Helper: set all P total flags ----------
inline void setAllPressureTot(OutputTagConfig& cfg) {
    cfg.writePXXTot = true;
    cfg.writePXYTot = true;
    cfg.writePXZTot = true;
    cfg.writePYYTot = true;
    cfg.writePYZTot = true;
    cfg.writePZZTot = true;
}

inline void addAllHeatFluxSpecies(OutputTagConfig& cfg, int ns) {
    for (auto& species : cfg.heatFluxSpecies)
        addAllSpecies(species, ns);
}

inline void addHeatFluxForSpecies(OutputTagConfig& cfg, int si) {
    for (auto& species : cfg.heatFluxSpecies)
        species.insert(si);
}

inline void setAllHeatFluxTot(OutputTagConfig& cfg) {
    for (auto& doWrite : cfg.writeHeatFluxTot)
        doWrite = true;
}

inline int heatFluxComponentFromToken(const std::string& token) {
    for (int c = 0; c < HeatFlux::ComponentCount; ++c)
        if (token == HeatFlux::ComponentNames[c]) return c;
    return -1;
}

/**
 * @brief Parse one moments-tag token.
 *
 * @return true if the token was recognised and handled, false otherwise.
 */
inline bool parseMomentToken(const std::string& tok, int ns,
                             OutputTagConfig& cfg)
{
    // --- Totals (check _tot before bare names) ---
    if (tok == "rho_tot") { cfg.writeRhoTot = true;  return true; }
    if (tok == "J_tot")   { cfg.writeJTot   = true;  return true; }
    if (tok == "P_tot")   { setAllPressureTot(cfg);   return true; }
    if (tok == "PXX_tot") { cfg.writePXXTot = true;   return true; }
    if (tok == "PXY_tot") { cfg.writePXYTot = true;   return true; }
    if (tok == "PXZ_tot") { cfg.writePXZTot = true;   return true; }
    if (tok == "PYY_tot") { cfg.writePYYTot = true;   return true; }
    if (tok == "PYZ_tot") { cfg.writePYZTot = true;   return true; }
    if (tok == "PZZ_tot") { cfg.writePZZTot = true;   return true; }
    if (tok == "Q_tot")   { setAllHeatFluxTot(cfg);   return true; }
    for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
        if (tok == std::string(HeatFlux::ComponentNames[c]) + "_tot") {
            cfg.writeHeatFluxTot[c] = true;
            return true;
        }
    }

    // --- Bare names (all species) ---
    if (tok == "rho") { addAllSpecies(cfg.rhoSpecies, ns);  return true; }
    if (tok == "J")   { addAllSpecies(cfg.JSpecies, ns);    return true; }
    if (tok == "P")   { addAllPressureSpecies(cfg, ns);     return true; }
    if (tok == "PXX") { addAllSpecies(cfg.PXXSpecies, ns);  return true; }
    if (tok == "PXY") { addAllSpecies(cfg.PXYSpecies, ns);  return true; }
    if (tok == "PXZ") { addAllSpecies(cfg.PXZSpecies, ns);  return true; }
    if (tok == "PYY") { addAllSpecies(cfg.PYYSpecies, ns);  return true; }
    if (tok == "PYZ") { addAllSpecies(cfg.PYZSpecies, ns);  return true; }
    if (tok == "PZZ") { addAllSpecies(cfg.PZZSpecies, ns);  return true; }
    if (tok == "Q")   { addAllHeatFluxSpecies(cfg, ns);     return true; }
    {
        const int component = heatFluxComponentFromToken(tok);
        if (component >= 0) {
            addAllSpecies(cfg.heatFluxSpecies[component], ns);
            return true;
        }
    }

    // --- Species-indexed: name followed by one or more digits ---
    static const std::regex re("^(rho|J|PXX|PXY|PXZ|PYY|PYZ|PZZ|P|Qxxx|Qxxy|Qxxz|Qxyy|Qxyz|Qxzz|Qyyy|Qyyz|Qyzz|Qzzz|Q)(\\d+)$");
    std::smatch m;
    if (std::regex_match(tok, m, re)) {
        int si = std::stoi(m[2].str());
        if (si < 0 || si >= ns) {
            fprintf(stderr, "WARNING: MomentsOutputTag token '%s' references "
                    "species %d but ns=%d — ignored.\n", tok.c_str(), si, ns);
            return true; // consumed but invalid
        }
        const std::string base = m[1].str();
        if      (base == "rho") cfg.rhoSpecies.insert(si);
        else if (base == "J")   cfg.JSpecies.insert(si);
        else if (base == "P")   addPressureForSpecies(cfg, si);
        else if (base == "PXX") cfg.PXXSpecies.insert(si);
        else if (base == "PXY") cfg.PXYSpecies.insert(si);
        else if (base == "PXZ") cfg.PXZSpecies.insert(si);
        else if (base == "PYY") cfg.PYYSpecies.insert(si);
        else if (base == "PYZ") cfg.PYZSpecies.insert(si);
        else if (base == "PZZ") cfg.PZZSpecies.insert(si);
        else if (base == "Q")   addHeatFluxForSpecies(cfg, si);
        else {
            const int component = heatFluxComponentFromToken(base);
            if (component >= 0)
                cfg.heatFluxSpecies[component].insert(si);
        }
        return true;
    }

    return false; // not recognised
}

/**
 * @brief Parse a FieldOutputTag token (legacy backward compat included).
 *
 * @return true if the token was recognised.
 */
inline bool parseFieldToken(const std::string& tok, int ns,
                            OutputTagConfig& cfg)
{
    if (tok == "B") { cfg.writeB = true; return true; }
    if (tok == "E") { cfg.writeE = true; return true; }

    // --- Legacy migration: old field-tag tokens that carried species data ---
    if (tok == "Je")  { cfg.JSpecies.insert(0); return true; }
    if (tok == "Ji")  { cfg.JSpecies.insert(1); return true; }
    if (tok == "Je2") { cfg.JSpecies.insert(2); return true; }
    if (tok == "Ji3") { cfg.JSpecies.insert(3); return true; }
    if (tok == "rho") { cfg.writeRhoTot = true;  return true; }

    return false;
}

/**
 * @brief Split a tag string on '+' and return trimmed tokens.
 */
inline std::vector<std::string> splitTags(const std::string& s) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, '+')) {
        // trim whitespace
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a != std::string::npos)
            tokens.push_back(tok.substr(a, b - a + 1));
    }
    return tokens;
}

/**
 * @brief Parse both tag strings into a single OutputTagConfig.
 *
 * @param fieldTag   Raw FieldOutputTag string from input file.
 * @param momentsTag Raw MomentsOutputTag string from input file.
 * @param ns         Number of particle species.
 */
inline OutputTagConfig parseOutputTags(const std::string& fieldTag,
                                       const std::string& momentsTag,
                                       int ns)
{
    OutputTagConfig cfg;

    for (const auto& tok : splitTags(fieldTag)) {
        if (!parseFieldToken(tok, ns, cfg)) {
            fprintf(stderr, "WARNING: unrecognised FieldOutputTag token '%s' — ignored.\n",
                    tok.c_str());
        }
    }

    for (const auto& tok : splitTags(momentsTag)) {
        if (!parseMomentToken(tok, ns, cfg)) {
            fprintf(stderr, "WARNING: unrecognised MomentsOutputTag token '%s' — ignored.\n",
                    tok.c_str());
        }
    }

    return cfg;
}

#endif // OUTPUT_TAG_CONFIG_H
