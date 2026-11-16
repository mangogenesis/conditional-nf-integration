// Minimal C ABI over the LHAPDF C++ API, used by the local Python shim.
#include "LHAPDF/LHAPDF.h"

#include <cstddef>
#include <string>

#define LHAPDF_CTYPES_API __declspec(dllexport)

extern "C" {

LHAPDF_CTYPES_API void* lhapdf_mkpdf(const char* name, int member) {
  try {
    return static_cast<void*>(
        LHAPDF::mkPDF(std::string(name), static_cast<std::size_t>(member)));
  } catch (...) {
    return nullptr;
  }
}

LHAPDF_CTYPES_API double lhapdf_xfxQ(void* pdf, int id, double x, double q2) {
  if (pdf == nullptr) {
    return 0.0;
  }
  return static_cast<LHAPDF::PDF*>(pdf)->xfxQ(id, x, q2);
}

LHAPDF_CTYPES_API double lhapdf_alphasQ(void* pdf, double q) {
  if (pdf == nullptr) {
    return 0.0;
  }
  return static_cast<LHAPDF::PDF*>(pdf)->alphasQ(q);
}

LHAPDF_CTYPES_API void lhapdf_delete(void* pdf) {
  delete static_cast<LHAPDF::PDF*>(pdf);
}

}  // extern "C"
