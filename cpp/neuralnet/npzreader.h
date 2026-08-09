#ifndef NPZREADER_H
#define NPZREADER_H

#include "../core/global.h"

#include <map>
#include <set>
#include <string>
#include <vector>

// Minimal reader for .npz archives produced by numpy.savez / savez_compressed.
// Only the subset of npy headers used by the accuracy corpus is supported:
// little-endian f4/f8/i2/u1 arrays, C (fortran_order=False) layout, one or two
// dimensions with non-negative sizes. Compression method 8 (zlib deflate) and
// method 0 (stored) are supported.
struct NpzArray {
  std::vector<int64_t> shape;
  std::string dtype;      // normalized: "f4","f8","i2","u1",...
  size_t elemSize = 0;
  std::vector<unsigned char> data;
  size_t numElements() const {
    size_t n = 1;
    for(int64_t s : shape)
      n *= (size_t)s;
    return n;
  }
};

struct NpzReader {
  std::map<std::string, NpzArray> arrays;

  explicit NpzReader(const std::string& fileName);
  NpzReader(const std::string& fileName, const std::set<std::string>& requestedNames);
  bool has(const std::string& name) const { return arrays.find(name) != arrays.end(); }
  const NpzArray& get(const std::string& name) const {
    auto it = arrays.find(name);
    if(it == arrays.end())
      throw StringError("npz: missing array " + name);
    return it->second;
  }
};

#endif
