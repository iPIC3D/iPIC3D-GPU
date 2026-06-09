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
};

class RestartSlotManager {
public:
    void init(const std::string& rootDir, const std::string& backend,
              int rank, int nranks, const RestartMeshMetadata& mesh);

    RestartWriteTarget beginWrite(int cycle);
    void publishIfRoot(const RestartWriteTarget& target) const;
    void completeLocal(const RestartWriteTarget& target);

    static RestartCheckpoint resolveLatest(const std::string& rootDir,
                                           const std::string& backend);
    static RestartCheckpoint readManifest(const std::string& dataDir,
                                          const std::string& backend);

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
    RestartMeshMetadata mesh_;
};

#endif // RESTART_SLOT_MANAGER_H
