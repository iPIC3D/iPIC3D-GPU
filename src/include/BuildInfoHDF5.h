/**
 * @file BuildInfoHDF5.h
 * @brief Write the build identity (BuildInfo.h) as string attributes
 *        "ipic3d_version" and "ipic3d_build" of an HDF5 file's root group.
 *
 * With parallel HDF5 every rank calls this with the same strings (attribute
 * creation is collective).
 */
#ifndef IPIC3D_BUILD_INFO_HDF5_H
#define IPIC3D_BUILD_INFO_HDF5_H

#include "BuildInfo.h"
#include "hdf5.h"

#include <string>

namespace iPic3D {
namespace build {

inline void writeHDF5Attributes(hid_t file) {
  const std::string values[2] = {version(), signature()};
  const char* names[2] = {"ipic3d_version", "ipic3d_build"};
  hid_t space = H5Screate(H5S_SCALAR);
  for (int i = 0; i < 2; i++) {
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, values[i].size() + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    hid_t attr =
        H5Acreate2(file, names[i], type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0) {
      H5Awrite(attr, type, values[i].c_str());
      H5Aclose(attr);
    }
    H5Tclose(type);
  }
  H5Sclose(space);
}

} // namespace build
} // namespace iPic3D

#endif // IPIC3D_BUILD_INFO_HDF5_H
