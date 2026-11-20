#ifndef OUTPUT_PREPARE_H
#define OUTPUT_PREPARE_H

#include <string>
#include <vector>

int checkOutputFolder(std::string relativePath);
int ensureOutputFolder(std::string relativePath);
/** Rename the files that exist to <stem>_before_cycle<cycle><extension>,
 *  or with _2, _3, ... after the cycle when a file of that name exists, the
 *  same suffix for all (a JSON and its binary): records that a restart does
 *  not continue are kept, never overwritten. */
void keepAside(const std::vector<std::string>& paths, int cycle);

// ======= Records of the data analysis =======
/** Add `"cycle": record` to the JSON object of the file, creating it. A
 *  record is a JSON object without braces in its strings; it starts a line
 *  with `"cycle": `, and no other line starts with a quote and a digit. */
void appendJsonRecord(const std::string& path, int cycle,
                      const std::string& record);
/** Remove the records of appendJsonRecord from cycle firstCycle on, which a
 *  restart writes again, and the part of an append cut short (a missing file
 *  is left so, one without records removed). */
void dropJsonRecordsFrom(const std::string& path, int firstCycle);
/** Remove the files <name>_<cycle>.vtk of the folder from cycle firstCycle
 *  on, which a restart writes again. */
void dropVtkFrom(const std::string& folder, int firstCycle);

#endif // OUTPUT_PREPARE_H
