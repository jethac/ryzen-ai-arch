// Diagnostic interposer only: capture buffers at the XRT sync boundary.
// Build: c++ -shared -fPIC -O2 trace-xrt-buffers.cpp -ldl -o trace-xrt-buffers.so
// Set XRT_TRACE_DIR and LD_PRELOAD only for the diagnostic Python process.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <string>

static void* resolve(const char* name) {
  static void* lib = dlopen("libxrt_coreutil.so.2", RTLD_NOW | RTLD_NOLOAD);
  if (!lib) lib = dlopen("libxrt_coreutil.so.2", RTLD_NOW);
  void* symbol = lib ? dlsym(lib, name) : nullptr;
  if (!symbol) { std::fprintf(stderr, "XRT trace: missing %s\n", name); std::abort(); }
  return symbol;
}

extern "C" void trace_sync(void*, int, size_t, size_t)
  asm("_ZN3xrt2bo4syncE18xclBOSyncDirectionmm");
extern "C" void trace_sync(void* self, int direction, size_t size, size_t offset) {
  using Sync = void (*)(void*, int, size_t, size_t);
  using Map = void* (*)(void*);
  using Size = size_t (*)(const void*);
  static auto sync = reinterpret_cast<Sync>(resolve("_ZN3xrt2bo4syncE18xclBOSyncDirectionmm"));
  static auto map = reinterpret_cast<Map>(resolve("_ZN3xrt2bo3mapEv"));
  static auto bo_size = reinterpret_cast<Size>(resolve("_ZNK3xrt2bo4sizeEv"));
  static std::atomic<unsigned> sequence{0};
  sync(self, direction, size, offset);
  const char* directory = std::getenv("XRT_TRACE_DIR");
  if (!directory) return;
  auto total = bo_size(self);
  if (offset > total || size > total-offset) std::abort();
  char name[160];
  std::snprintf(name, sizeof(name), "/%04u-dir%d-size%zu-offset%zu.bin",
                sequence++, direction, size, offset);
  std::string path = std::string(directory) + name;
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) { std::perror(path.c_str()); std::abort(); }
  auto data = static_cast<unsigned char*>(map(self));
  if (std::fwrite(data+offset, 1, size, file) != size) std::abort();
  std::fclose(file);
  std::fprintf(stderr, "XRT_TRACE %s bo_size=%zu\n", path.c_str(), total);
}
