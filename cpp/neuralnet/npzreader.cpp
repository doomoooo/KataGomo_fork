#include "../neuralnet/npzreader.h"

#include <zlib.h>

#include <cstring>
#include <fstream>

using namespace std;

namespace {

uint16_t readU16(const vector<unsigned char>& b, size_t off) {
  if(off + 2 > b.size())
    throw StringError("npz: truncated uint16");
  return (uint16_t)b[off] | ((uint16_t)b[off+1] << 8);
}

uint32_t readU32(const vector<unsigned char>& b, size_t off) {
  if(off + 4 > b.size())
    throw StringError("npz: truncated uint32");
  return (uint32_t)b[off] | ((uint32_t)b[off+1] << 8) |
         ((uint32_t)b[off+2] << 16) | ((uint32_t)b[off+3] << 24);
}

uint64_t readU64(const vector<unsigned char>& b, size_t off) {
  if(off + 8 > b.size())
    throw StringError("npz: truncated uint64");
  uint64_t v = 0;
  for(int i = 7; i >= 0; i--)
    v = (v << 8) | b[off + (size_t)i];
  return v;
}

struct CentralEntry {
  string name;
  uint16_t method;
  uint32_t compressedSize;
  uint32_t uncompressedSize;
  uint64_t localHeaderOffset;
};

string parseDtype(const string& descr, size_t* elemSize) {
  // numpy descr strings we support: '|u1', '|i1', '<i2', '<f4', '<f8', '=f4' etc.
  size_t pos = 0;
  while(pos < descr.size() && (descr[pos] == '\'' || descr[pos] == '"' || descr[pos] == ' '))
    pos++;
  string t;
  for(size_t i = pos; i < descr.size(); i++) {
    char c = descr[i];
    if(c == '\'' || c == '"' || c == ' ')
      break;
    t += c;
  }
  if(t.size() < 2)
    throw StringError("npz: unsupported dtype descr '" + descr + "'");
  char endian = t[0];
  if(endian != '<' && endian != '|' && endian != '=')
    throw StringError("npz: unsupported dtype endian '" + descr + "'");
  char kind = t[1];
  size_t size = 0;
  if(t.size() >= 3)
    size = (size_t)(t[2] - '0');
  if(t.size() >= 4)
    size = size * 10 + (size_t)(t[3] - '0');
  if(size == 0)
    throw StringError("npz: unsupported dtype descr '" + descr + "'");
  string norm;
  if(kind == 'f' && size == 4) norm = "f4";
  else if(kind == 'f' && size == 8) norm = "f8";
  else if(kind == 'i' && size == 2) norm = "i2";
  else if(kind == 'u' && size == 1) norm = "u1";
  else if(kind == 'i' && size == 1) norm = "i1";
  else
    throw StringError("npz: unsupported dtype '" + descr + "'");
  if(elemSize != NULL)
    *elemSize = size;
  return norm;
}

void parseNpyHeader(
  const vector<unsigned char>& payload,
  NpzArray& out
) {
  if(payload.size() < 10 || payload[0] != 0x93 || payload[1] != 'N' ||
     payload[2] != 'U' || payload[3] != 'M' || payload[4] != 'P' || payload[5] != 'Y')
    throw StringError("npz: array is not an npy payload");
  uint16_t version = readU16(payload, 6);
  size_t headerLen = 0;
  size_t headerStart = 0;
  if(version == 1) {
    headerLen = readU16(payload, 8);
    headerStart = 10;
  }
  else if(version == 2) {
    headerLen = (size_t)readU32(payload, 8);
    headerStart = 12;
  }
  else
    throw StringError("npz: unsupported npy version " + Global::intToString(version));
  if(headerStart + headerLen > payload.size())
    throw StringError("npz: npy header truncated");
  string header((const char*)&payload[headerStart], headerLen);

  size_t descrPos = header.find("'descr'");
  if(descrPos == string::npos)
    throw StringError("npz: npy header missing descr");
  size_t colon = header.find(':', descrPos);
  size_t q1 = header.find('\'', colon + 1);
  size_t q2 = header.find('\'', q1 + 1);
  if(q1 == string::npos || q2 == string::npos)
    throw StringError("npz: npy header malformed descr");
  string descr = header.substr(q1 + 1, q2 - q1 - 1);
  out.dtype = parseDtype(descr, &out.elemSize);

  size_t shapePos = header.find("'shape'");
  if(shapePos == string::npos)
    throw StringError("npz: npy header missing shape");
  size_t openParen = header.find('(', shapePos);
  size_t closeParen = header.find(')', openParen);
  if(openParen == string::npos || closeParen == string::npos)
    throw StringError("npz: npy header malformed shape");
  string shapeStr = header.substr(openParen + 1, closeParen - openParen - 1);
  out.shape.clear();
  size_t pos = 0;
  while(pos <= shapeStr.size()) {
    size_t comma = shapeStr.find(',', pos);
    string token = shapeStr.substr(pos, comma == string::npos ? string::npos : comma - pos);
    while(!token.empty() && (token[0] == ' ' || token[0] == '\t'))
      token.erase(token.begin());
    while(!token.empty() && (token[token.size()-1] == ' ' || token[token.size()-1] == '\t'))
      token.erase(token.end()-1);
    if(!token.empty())
      out.shape.push_back(Global::stringToInt64(token));
    if(comma == string::npos)
      break;
    pos = comma + 1;
  }
  if(out.shape.empty())
    out.shape.push_back(0);

  size_t dataStart = headerStart + headerLen;
  size_t expected = out.numElements() * out.elemSize;
  if(dataStart + expected > payload.size())
    throw StringError("npz: array payload shorter than header declares");
  out.data.assign(payload.begin() + (ptrdiff_t)dataStart, payload.begin() + (ptrdiff_t)(dataStart + expected));
}

}  // namespace

NpzReader::NpzReader(const string& fileName) {
  ifstream in(fileName, ios::binary);
  if(!in)
    throw StringError("npz: could not open " + fileName);
  vector<unsigned char> whole((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
  if(whole.size() < 22)
    throw StringError("npz: file too small: " + fileName);
  uint32_t eocd = 0x06054b50;
  size_t eocdPos = string::npos;
  for(size_t i = whole.size() - 22 + 1; i-- > 0; ) {
    if(whole[i] == 0x50 && readU32(whole, i) == eocd) {
      eocdPos = i;
      break;
    }
  }
  if(eocdPos == string::npos)
    throw StringError("npz: no end-of-central-directory record in " + fileName);
  uint16_t count = readU16(whole, eocdPos + 10);
  uint32_t cdOffset = readU32(whole, eocdPos + 16);

  vector<CentralEntry> entries;
  size_t pos = cdOffset;
  for(int i = 0; i < count; i++) {
    if(pos + 46 > whole.size() || readU32(whole, pos) != 0x02014b50)
      throw StringError("npz: malformed central directory");
    CentralEntry e;
    uint16_t nameLen = readU16(whole, pos + 28);
    uint16_t extraLen = readU16(whole, pos + 30);
    uint16_t commentLen = readU16(whole, pos + 32);
    e.method = readU16(whole, pos + 10);
    e.compressedSize = readU32(whole, pos + 20);
    e.uncompressedSize = readU32(whole, pos + 24);
    e.localHeaderOffset = readU32(whole, pos + 42);
    if(pos + 46 + nameLen + extraLen + commentLen > whole.size())
      throw StringError("npz: central directory entry truncated");
    e.name.assign((const char*)&whole[pos + 46], nameLen);
    entries.push_back(e);
    pos += 46 + nameLen + extraLen + commentLen;
  }

  for(const CentralEntry& e : entries) {
    if(e.localHeaderOffset + 30 > whole.size() || readU32(whole, e.localHeaderOffset) != 0x04034b50)
      throw StringError("npz: malformed local header for " + e.name);
    uint16_t nameLen = readU16(whole, e.localHeaderOffset + 26);
    uint16_t extraLen = readU16(whole, e.localHeaderOffset + 28);
    size_t dataStart = (size_t)e.localHeaderOffset + 30 + nameLen + extraLen;
    if(dataStart + e.compressedSize > whole.size())
      throw StringError("npz: local data truncated for " + e.name);
    vector<unsigned char> payload;
    if(e.method == 0) {
      payload.assign(whole.begin() + (ptrdiff_t)dataStart, whole.begin() + (ptrdiff_t)(dataStart + e.compressedSize));
    }
    else if(e.method == 8) {
      payload.resize(e.uncompressedSize);
      z_stream strm;
      memset(&strm, 0, sizeof(strm));
      if(inflateInit2(&strm, -MAX_WBITS) != Z_OK)
        throw StringError("npz: inflateInit failed for " + e.name);
      strm.next_in = (Bytef*)&whole[dataStart];
      strm.avail_in = e.compressedSize;
      strm.next_out = payload.data();
      strm.avail_out = (uInt)payload.size();
      int ret = inflate(&strm, Z_FINISH);
      int initRet = inflateEnd(&strm);
      if(ret != Z_STREAM_END || initRet != Z_OK)
        throw StringError("npz: inflate failed for " + e.name);
    }
    else {
      throw StringError("npz: unsupported compression method " + Global::intToString(e.method) + " for " + e.name);
    }
    if(e.name.size() >= 4 && e.name.compare(e.name.size() - 4, 4, ".npy") == 0) {
      NpzArray arr;
      parseNpyHeader(payload, arr);
      arrays[e.name.substr(0, e.name.size() - 4)] = std::move(arr);
    }
  }
}
