#include "outputPrepare.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void remove_all_contents(const fs::path& dir) {
  for (const auto& entry : fs::directory_iterator(dir)) {
    fs::remove_all(entry);
  }
}

/**
 * @brief prepare the output folder, create if needed
 *
 * @details Check if the output folder exists,
 *          if not, create it. Make sure it's there in the end.
 *          Fail with exception.
 *
 * @param relativePath: Output folder path relative to the current directory
 * @returns 0 if succeed
 */
int checkOutputFolder(std::string relativePath) {

  fs::path current_path = fs::current_path();
  fs::path subdir = current_path / relativePath;

  if (fs::exists(subdir)) {
    if (fs::is_directory(subdir)) {

      if (!fs::is_empty(subdir)) {
        remove_all_contents(subdir);
      }

    } else
      throw std::runtime_error("Output path " + relativePath +
                               " is not a directory");
  } else {
    fs::create_directories(subdir);
  }

  return 0;
}

int ensureOutputFolder(std::string relativePath) {

  fs::path current_path = fs::current_path();
  fs::path subdir = current_path / relativePath;

  if (fs::exists(subdir)) {
    if (!fs::is_directory(subdir)) {
      throw std::runtime_error("Output path " + relativePath +
                               " is not a directory");
    }
  } else {
    fs::create_directories(subdir);
  }

  return 0;
}

void keepAside(const std::vector<std::string>& paths, int cycle) {
  auto aside = [cycle](const fs::path& file, int copy) {
    fs::path name = file;
    name.replace_filename(file.stem().string() + "_before_cycle" +
                          std::to_string(cycle) +
                          (copy > 1 ? "_" + std::to_string(copy) : "") +
                          file.extension().string());
    return name;
  };
  // a suffix free for every file: a later restart from the same cycle keeps
  // its records too
  int copy = 1;
  while (std::any_of(paths.begin(), paths.end(), [&](const std::string& path) {
    return fs::exists(aside(path, copy));
  }))
    copy++;
  for (const std::string& path : paths)
    if (fs::exists(path))
      fs::rename(path, aside(path, copy));
}

void appendJsonRecord(const std::string& path, int cycle,
                      const std::string& record) {
  if (!fs::exists(path))
    std::ofstream(path).close();
  std::fstream out(path, std::ios::in | std::ios::out | std::ios::ate);
  if (!out.is_open())
    throw std::runtime_error("[!]Error: Can not open the output file " + path);
  std::ostringstream buffer;
  if (out.tellg() == 0) {
    buffer << "{\n";
  } else {
    out.seekp(-2, std::ios_base::end); // over the closing "\n}"
    buffer << ",\n";
  }
  buffer << "\"" << cycle << "\": " << record << "\n}";
  out << buffer.str();
  out.close();
  if (!out)
    throw std::runtime_error("[!]Error: Can not write the output file " + path);
}

void dropJsonRecordsFrom(const std::string& path, int firstCycle) {
  std::ifstream in(path);
  if (!in.is_open())
    return;
  const std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
  in.close();
  // the records, at the lines that start with a quote and a digit
  std::vector<size_t> start;
  for (size_t line = 0; line + 1 < content.size();) {
    if (content[line] == '"' && std::isdigit((unsigned char)content[line + 1]))
      start.push_back(line);
    const size_t next = content.find('\n', line);
    if (next == std::string::npos)
      break;
    line = next + 1;
  }
  // the end of the records: that of the last one if its braces close, before
  // it if an append was cut short there (a full disk; its cycle, at or after
  // firstCycle, is written again too)
  size_t end = std::string::npos;
  if (!start.empty()) {
    int depth = 0;
    for (size_t i = content.find('{', start.back()); i < content.size(); i++)
      if (content[i] == '{')
        depth++;
      else if (content[i] == '}' && --depth == 0) {
        end = i + 1;
        break;
      }
    if (end == std::string::npos) {
      end = start.back() - 2; // before its ",\n"
      start.pop_back();
    }
  }
  // the records kept: before the first from firstCycle on
  size_t kept = 0;
  while (kept < start.size() &&
         std::atoi(content.c_str() + start[kept] + 1) < firstCycle)
    kept++;
  if (kept == 0) { // none
    fs::remove(path);
    return;
  }
  if (kept < start.size())
    end = start[kept] - 2; // before its ",\n"
  if (content.compare(end, std::string::npos, "\n}") == 0)
    return; // all kept, the file complete
  // written aside first: a failure leaves the records as they were
  const std::string temporary = path + ".tmp";
  std::ofstream out(temporary, std::ios::trunc);
  out << content.substr(0, end) << "\n}";
  out.close();
  if (!out)
    throw std::runtime_error("[!]Error: Can not write the output file " +
                             temporary);
  fs::rename(temporary, path);
}

void dropVtkFrom(const std::string& folder, int firstCycle) {
  std::vector<fs::path> drop;
  for (const auto& entry : fs::directory_iterator(folder)) {
    const std::string name = entry.path().filename().string();
    const size_t cycle = name.rfind('_') + 1; // 0 without '_'
    if (entry.path().extension() == ".vtk" && cycle > 0 &&
        std::isdigit((unsigned char)name[cycle]) &&
        std::atoi(name.c_str() + cycle) >= firstCycle)
      drop.push_back(entry.path());
  }
  for (const auto& path : drop)
    fs::remove(path);
}
