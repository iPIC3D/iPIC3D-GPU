#ifndef RESTART_SLOT_MANAGER_H
#define RESTART_SLOT_MANAGER_H

#include "RestartMeshMetadata.h"

#include <string>

struct RestartCheckpoint {
  bool found = false;
  std::string rootDir;
  std::string dataDir;
  std::string slot;
  std::string backend;
  int cycle = -1;
  int nranks = 0;
  long long generation = 0;
  RestartMeshMetadata mesh;
  /** Build that wrote the checkpoint (BuildInfo.h); empty if not recorded. */
  std::string buildVersion;
  std::string buildSignature;
};

struct RestartWriteTarget {
  std::string rootDir;
  std::string dataDir;
  std::string slot;
  std::string backend;
  std::string rankFilePath;
  int cycle = -1;
  int rank = 0;
  int nranks = 1;
  long long generation = 1;
  RestartCheckpoint previous;
  RestartMeshMetadata mesh;
  /** Build writing the checkpoint (BuildInfo.h). */
  std::string buildVersion;
  std::string buildSignature;
};

/**
 * @brief The two checkpoint slots (restart_A, restart_B) under a restart root.
 *
 * Writing: each checkpoint goes to the slot that does not hold the checkpoint
 * the run continues from. Rank 0 first removes that slot's manifest.json, so
 * a slot whose rewrite was cut off is never taken for a complete checkpoint;
 * after every rank has closed its new rank file, rank 0 writes the manifest
 * and latest_restart.json.
 *
 * Reading: a slot is usable when its manifest.json belongs to it, matches the
 * restart backend, has a valid cycle, rank count and mesh, and a rank file
 * exists for each of its nranks ranks.
 */
class RestartSlotManager {
public:
  /**
   * @param restartSlot slot ("A" or "B") this run was restarted from, or ""
   *        for a fresh run; the first checkpoint goes to the other slot.
   */
  void init(const std::string& rootDir, const std::string& backend, int rank,
            int nranks, const RestartMeshMetadata& mesh,
            const std::string& restartSlot = "");

  /** Choose the slot and generation of a new checkpoint; rank 0 removes the
   *  slot's manifest. Call on every rank, then barrier before
   *  prepareRankFile(). */
  RestartWriteTarget beginWrite(int cycle);
  /** Remove this rank's file from the previous use of the slot. */
  void prepareRankFile(const RestartWriteTarget& target) const;
  void publishIfRoot(const RestartWriteTarget& target) const;
  void completeLocal(const RestartWriteTarget& target);

  /**
   * @brief Select the checkpoint to restart from.
   *
   * requestedSlot "A" or "B" tries that slot, then the other. "" tries the
   * slot latest_restart.json names as latest, then the other; without that
   * file, the slot whose manifest has the higher generation first. The first
   * usable slot is returned; `report` explains every slot that was passed
   * over and, when the other slot holds a newer usable checkpoint, that the
   * next checkpoint will overwrite it. Returns found == false if no slot is
   * usable.
   */
  static RestartCheckpoint resolve(const std::string& rootDir,
                                   const std::string& backend,
                                   const std::string& requestedSlot,
                                   std::string& report);
  /**
   * @brief Load one slot from its manifest.json.
   * @param checkRankFiles also require every rank file to exist.
   * @return found == false, with `reason` set, if the slot is not usable.
   */
  static RestartCheckpoint loadSlot(const std::string& rootDir,
                                    const std::string& backend,
                                    const std::string& slot,
                                    bool checkRankFiles, std::string& reason);
  static RestartCheckpoint readManifest(const std::string& dataDir,
                                        const std::string& backend);

  /** "A"/"a" -> "A", "B"/"b" -> "B", "latest" or "" -> ""; false otherwise. */
  static bool parseSlotName(const std::string& text, std::string& slot);
  static std::string backendName();
  static std::string slotDir(const std::string& rootDir,
                             const std::string& slot);
  static std::string rankFileName(const std::string& backend, int rank);

private:
  std::string rootDir_;
  std::string backend_;
  int rank_ = 0;
  int nranks_ = 1;
  bool initialized_ = false;
  RestartCheckpoint latest_;
  /** Highest generation found in the restart root; new ones exceed it. */
  long long maxGeneration_ = 0;
  RestartMeshMetadata mesh_;
};

#endif // RESTART_SLOT_MANAGER_H
