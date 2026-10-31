#include "vpipelinelogger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <atomic>
#include <mutex>
#include <string>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__ANDROID__)
#include <android/log.h>
#define LOG_TAG "OpenGothic"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) printf(__VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif

namespace {
struct Config {
#if defined(__EMSCRIPTEN__)
  // WebGPU: no binding arrays, no image atomics (Hi-Z mip chain needs them)
  int no_bindless     = 1;
  int no_dynrendering = 0;
  int no_multidraw    = 1;
  int skip_hiz        = 1;
#else
  int no_bindless     = 0;
  int no_dynrendering = 0;
  int no_multidraw    = 0;
  int skip_hiz        = 0;
#endif
  int skip_shadows    = 0;
  int skip_lights     = 0;
  int skip_fog        = 0;
  int skip_gbuf       = 0;
  int skip_atest      = 0;
  int always_stash    = 0; // copy the scene for water/glass even when nothing in the frame reads it
  int no_cmd_cull     = 0; // do not skip the draw commands that are wholly outside of the camera
  int spirv_noopt     = 0;
#if defined(__ANDROID__)
  // solid-depth variants for alpha-test in shadow/depth passes: no discard FS, LRZ-friendly on Adreno
  int no_atest_depth  = 1;
#else
  int no_atest_depth  = 0;
#endif
  int force_hiz       = 0;
  int prewarm         = 0;
  int log_pipelines   = 1;
  int safe_mode       = 0;
  int render_scale    = 100;
  int simple_render   = 0;
  int draw_dist       = 0; // objects: 0 = no limit, 1..4 = 150/80/40/20 m
  int world_dist_m    = 0; // everything: meters, 0 = no limit
  int shadows         = 1; // android: 0 = off, 1 = 512, 2 = 1024, 3 = 2048
  int ssao            = 0; // android: ambient occlusion (standard render only)
  int reflections     = 0; // android: screen-space water reflections
  int show_fps        = 1;
  int fps_unlimited   = 0; // 1 = no game fps limit (zMaxFps)
  int freeze_time     = 0; // hour 1..24 to pin time-of-day for stable benchmarking; 0 = off
  bool loaded         = false;
};

static Config s_config;
static std::mutex s_logMutex;
static std::atomic<uint32_t> s_pipelineSeq{1};

static std::string s_gameDirOverride;

static const char* findGameDir() {
  if(!s_gameDirOverride.empty())
    return s_gameDirOverride.c_str();
  const char* candidates[] = {
    "/storage/emulated/0/Gothic2",
    "/sdcard/Gothic2",
    "/storage/emulated/0/Gothic",
    "/sdcard/Gothic"
  };
  for(auto path : candidates) {
    struct stat st;
    if(stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
      return path;
    }
  }
  return "/sdcard/Gothic2";
}

static void ensureDir(const char* dir) {
  mkdir(dir, 0777);
}

static void rawLogToFile(const char* filepath, const char* str) {
  FILE* f = fopen(filepath, "a");
  if(f) {
    fputs(str, f);
    fflush(f);
    int fd = fileno(f);
    if(fd >= 0) {
      fsync(fd);
    }
    fclose(f);
  }
}

static void logBoth(const char* msg) {
  LOGI("%s", msg);

  const char* dir = findGameDir();
  ensureDir(dir);

  char pipePath[512];
  snprintf(pipePath, sizeof(pipePath), "%s/pipelines.log", dir);
  rawLogToFile(pipePath, msg);

  char crashPath[512];
  snprintf(crashPath, sizeof(crashPath), "%s/crash.log", dir);
  rawLogToFile(crashPath, msg);
}
} // namespace

extern "C" {

static time_t s_lastCfgMtime = 0;
static std::chrono::steady_clock::time_point s_lastCfgCheck{};

static void parseConfigFile(const char* cfgPath, time_t mtime) {
  FILE* f = fopen(cfgPath, "r");
  if(!f) return;
  s_lastCfgMtime = mtime;

  char line[256];
  while(fgets(line, sizeof(line), f)) {
    if(line[0] == '#' || line[0] == ';' || line[0] == '/' || line[0] == '\n' || line[0] == '\r')
      continue;
    char key[128] = {0};
    int val = 0;
    if(sscanf(line, "%127[^=]=%d", key, &val) == 2) {
      char* k = key;
      while(*k == ' ' || *k == '\t') k++;
      int len = strlen(k);
      while(len > 0 && (k[len-1] == ' ' || k[len-1] == '\t' || k[len-1] == '\r' || k[len-1] == '\n')) {
        k[len-1] = '\0';
        len--;
      }

      if(strcmp(k, "no_bindless") == 0) s_config.no_bindless = val;
      else if(strcmp(k, "no_dynrendering") == 0) s_config.no_dynrendering = val;
      else if(strcmp(k, "no_multidraw") == 0) s_config.no_multidraw = val;
      else if(strcmp(k, "skip_hiz") == 0) s_config.skip_hiz = val;
      else if(strcmp(k, "skip_shadows") == 0) s_config.skip_shadows = val;
      else if(strcmp(k, "skip_lights") == 0) s_config.skip_lights = val;
      else if(strcmp(k, "skip_fog") == 0) s_config.skip_fog = val;
      else if(strcmp(k, "skip_gbuf") == 0) s_config.skip_gbuf = val;
      else if(strcmp(k, "skip_atest") == 0) s_config.skip_atest = val;
      else if(strcmp(k, "no_cmd_cull") == 0) s_config.no_cmd_cull = val;
      else if(strcmp(k, "always_stash") == 0) s_config.always_stash = val;
      else if(strcmp(k, "no_atest_depth") == 0) s_config.no_atest_depth = val;
      else if(strcmp(k, "prewarm") == 0) s_config.prewarm = val;
      else if(strcmp(k, "spirv_noopt") == 0) s_config.spirv_noopt = val;
      else if(strcmp(k, "log_pipelines") == 0) s_config.log_pipelines = val;
      else if(strcmp(k, "safe_mode") == 0) s_config.safe_mode = val;
      else if(strcmp(k, "render_scale") == 0) s_config.render_scale = val;
      else if(strcmp(k, "simple_render") == 0) s_config.simple_render = val;
      else if(strcmp(k, "freeze_time") == 0) s_config.freeze_time = val;
      else if(strcmp(k, "draw_dist") == 0) s_config.draw_dist = val;
      else if(strcmp(k, "world_dist") == 0) s_config.world_dist_m = val;
      else if(strcmp(k, "shadows") == 0) s_config.shadows = val;
      else if(strcmp(k, "ssao") == 0) s_config.ssao = val;
      else if(strcmp(k, "reflections") == 0) s_config.reflections = val;
      else if(strcmp(k, "show_fps") == 0) s_config.show_fps = val;
      else if(strcmp(k, "fps_unlimited") == 0) s_config.fps_unlimited = val;
      else if(strcmp(k, "force_hiz") == 0) s_config.force_hiz = val;
    }
  }
  fclose(f);

  if(s_config.safe_mode) {
    s_config.no_bindless     = 1;
    s_config.no_multidraw    = 1;
    s_config.skip_hiz        = 1;
    s_config.no_atest_depth  = 1;
  }
}

void opengothic_debug_check_reload() {
  std::lock_guard<std::mutex> lock(s_logMutex);
  auto now = std::chrono::steady_clock::now();
  if(std::chrono::duration_cast<std::chrono::milliseconds>(now - s_lastCfgCheck).count() < 300)
    return;
  s_lastCfgCheck = now;

  const char* dir = findGameDir();
  char cfgPath[512];
  snprintf(cfgPath, sizeof(cfgPath), "%s/debug.cfg", dir);
  struct stat st;
  if(stat(cfgPath, &st) == 0) {
    if(st.st_mtime != s_lastCfgMtime) {
      parseConfigFile(cfgPath, st.st_mtime);
      LOGI("debug.cfg reloaded: render_scale=%d, simple_render=%d", s_config.render_scale, s_config.simple_render);
    }
  }
}

void opengothic_debug_set_game_dir(const char* dir) {
  std::lock_guard<std::mutex> lock(s_logMutex);
  if(dir!=nullptr)
    s_gameDirOverride = dir;
  }

void opengothic_debug_init() {
  std::lock_guard<std::mutex> lock(s_logMutex);
  if(s_config.loaded) {
    opengothic_debug_check_reload();
    return;
  }
  s_config.loaded = true;

  const char* dir = findGameDir();
  ensureDir(dir);

  char cfgPath[512];
  snprintf(cfgPath, sizeof(cfgPath), "%s/debug.cfg", dir);

  struct stat st;
  if(stat(cfgPath, &st) == 0) {
    parseConfigFile(cfgPath, st.st_mtime);
  } else {
    // Create template debug.cfg for user convenience
    FILE* out = fopen(cfgPath, "w");
    if(out) {
      fputs("# OpenGothic Debug Configuration\n", out);
      fputs("# Set any option to 1 to enable, 0 to disable\n", out);
      fputs("safe_mode=0\n", out);
      fputs("no_bindless=0\n", out);
      fputs("no_dynrendering=0\n", out);
      fputs("no_multidraw=0\n", out);
      fputs("skip_hiz=0\n", out);
      fputs("skip_shadows=0\n", out);
      fputs("no_atest_depth=0\n", out);
      fputs("prewarm=0\n", out);
      fputs("spirv_noopt=0\n", out);
      fputs("log_pipelines=1\n", out);
      fputs("render_scale=100\n", out);
      fputs("simple_render=0\n", out);
      fflush(out);
      fclose(out);
    }
    if(stat(cfgPath, &st) == 0) {
      parseConfigFile(cfgPath, st.st_mtime);
    }
  }

  // Prewarm crash detection from previous run
  char curFile[512], crashFile[512];
  snprintf(curFile, sizeof(curFile), "%s/prewarm_current.txt", dir);
  snprintf(crashFile, sizeof(crashFile), "%s/prewarm_crashed.txt", dir);
  FILE* curF = fopen(curFile, "r");
  if(curF) {
    char crashedKey[256] = {0};
    if(fgets(crashedKey, sizeof(crashedKey), curF)) {
      char* nl = strpbrk(crashedKey, "\r\n");
      if(nl) *nl = '\0';
      if(crashedKey[0] != '\0') {
        char msg[512];
        snprintf(msg, sizeof(msg), "\n[PREWARM] CRASH IN PREVIOUS RUN DETECTED FOR: %s -> Recording to prewarm_crashed.txt\n", crashedKey);
        logBoth(msg);

        FILE* crOut = fopen(crashFile, "a");
        if(crOut) {
          fprintf(crOut, "%s\n", crashedKey);
          fflush(crOut);
          fclose(crOut);
        }
      }
    }
    fclose(curF);
    unlink(curFile);
  }

  char buf[512];
  snprintf(buf, sizeof(buf),
    "\n============================================================\n"
    "[DEBUG_CFG] Loaded from %s:\n"
    "  safe_mode       = %d\n"
    "  no_bindless     = %d\n"
    "  no_dynrendering = %d\n"
    "  no_multidraw    = %d\n"
    "  skip_hiz        = %d\n"
    "  skip_shadows    = %d\n"
    "  no_atest_depth  = %d\n"
    "  prewarm         = %d\n"
    "  spirv_noopt     = %d\n"
    "  log_pipelines   = %d\n"
    "============================================================\n",
    cfgPath, s_config.safe_mode, s_config.no_bindless, s_config.no_dynrendering,
    s_config.no_multidraw, s_config.skip_hiz, s_config.skip_shadows,
    s_config.no_atest_depth, s_config.prewarm,
    s_config.spirv_noopt, s_config.log_pipelines
  );
  logBoth(buf);
}

int  opengothic_debug_no_bindless()     { if(!s_config.loaded) opengothic_debug_init(); return s_config.no_bindless; }
int  opengothic_debug_no_dynrendering() { if(!s_config.loaded) opengothic_debug_init(); return s_config.no_dynrendering; }
int  opengothic_debug_no_multidraw()    { if(!s_config.loaded) opengothic_debug_init(); return s_config.no_multidraw; }
int  opengothic_debug_skip_hiz()        { if(!s_config.loaded) opengothic_debug_init(); return s_config.skip_hiz; }
int  opengothic_debug_skip_shadows()    { if(!s_config.loaded) opengothic_debug_init(); return s_config.skip_shadows; }
int  opengothic_debug_skip_lights()     { if(!s_config.loaded) opengothic_debug_init(); return s_config.skip_lights; }
int  opengothic_debug_skip_fog()        { if(!s_config.loaded) opengothic_debug_init(); return s_config.skip_fog; }
int  opengothic_debug_skip_gbuf()       { if(!s_config.loaded) opengothic_debug_init(); return s_config.skip_gbuf; }
int  opengothic_debug_always_stash()   { if(!s_config.loaded) opengothic_debug_init(); return s_config.always_stash; }
int  opengothic_debug_no_cmd_cull()    { if(!s_config.loaded) opengothic_debug_init(); return s_config.no_cmd_cull; }
int  opengothic_debug_skip_atest()      { if(!s_config.loaded) opengothic_debug_init(); return s_config.skip_atest; }
int  opengothic_debug_no_atest_depth()  { if(!s_config.loaded) opengothic_debug_init(); return s_config.no_atest_depth; }
int  opengothic_debug_prewarm()         { if(!s_config.loaded) opengothic_debug_init(); return s_config.prewarm; }
int  opengothic_debug_spirv_noopt()     { if(!s_config.loaded) opengothic_debug_init(); return s_config.spirv_noopt; }
int  opengothic_debug_log_pipelines()   { if(!s_config.loaded) opengothic_debug_init(); return s_config.log_pipelines; }
int  opengothic_debug_safe_mode()       { if(!s_config.loaded) opengothic_debug_init(); return s_config.safe_mode; }
int  opengothic_debug_freeze_time()     { if(!s_config.loaded) opengothic_debug_init(); else opengothic_debug_check_reload(); return s_config.freeze_time; }
int  opengothic_debug_draw_dist()     { if(!s_config.loaded) opengothic_debug_init(); return s_config.draw_dist; }
int  opengothic_debug_shadows()      { if(!s_config.loaded) opengothic_debug_init(); return s_config.shadows; }
int  opengothic_debug_ssao()         { if(!s_config.loaded) opengothic_debug_init(); return s_config.ssao; }
int  opengothic_debug_reflections()  { if(!s_config.loaded) opengothic_debug_init(); return s_config.reflections; }
int  opengothic_debug_show_fps()     { if(!s_config.loaded) opengothic_debug_init(); return s_config.show_fps; }
int  opengothic_debug_fps_unlimited(){ if(!s_config.loaded) opengothic_debug_init(); return s_config.fps_unlimited; }
int  opengothic_debug_world_dist()    { if(!s_config.loaded) opengothic_debug_init(); return s_config.world_dist_m; }
int  opengothic_debug_force_hiz()       { if(!s_config.loaded) opengothic_debug_init(); else opengothic_debug_check_reload(); return s_config.force_hiz; }
int  opengothic_debug_use_hiz_prepass() {
#if defined(__ANDROID__) || defined(__EMSCRIPTEN__)
  return opengothic_debug_force_hiz();
#else
  return 1;
#endif
  }
int  opengothic_debug_render_scale()    { if(!s_config.loaded) opengothic_debug_init(); else opengothic_debug_check_reload(); return s_config.render_scale; }
int  opengothic_debug_simple_render()   { if(!s_config.loaded) opengothic_debug_init(); else opengothic_debug_check_reload(); return s_config.simple_render; }

int opengothic_prewarm_check_crashed(const char* key) {
  const char* dir = findGameDir();
  char crashFile[512];
  snprintf(crashFile, sizeof(crashFile), "%s/prewarm_crashed.txt", dir);
  FILE* f = fopen(crashFile, "r");
  if(!f) return 0;
  char line[256];
  int found = 0;
  while(fgets(line, sizeof(line), f)) {
    char* nl = strpbrk(line, "\r\n");
    if(nl) *nl = '\0';
    if(strcmp(line, key) == 0) {
      found = 1;
      break;
    }
  }
  fclose(f);
  return found;
}

void opengothic_prewarm_begin_attempt(const char* key) {
  const char* dir = findGameDir();
  ensureDir(dir);
  char curFile[512];
  snprintf(curFile, sizeof(curFile), "%s/prewarm_current.txt", dir);
  FILE* f = fopen(curFile, "w");
  if(f) {
    fprintf(f, "%s\n", key);
    fflush(f);
    int fd = fileno(f);
    if(fd >= 0) fsync(fd);
    fclose(f);
  }
}

void opengothic_prewarm_end_attempt(const char* key, int ok) {
  const char* dir = findGameDir();
  char curFile[512];
  snprintf(curFile, sizeof(curFile), "%s/prewarm_current.txt", dir);
  unlink(curFile);

  if(!ok) {
    char crashFile[512];
    snprintf(crashFile, sizeof(crashFile), "%s/prewarm_crashed.txt", dir);
    FILE* crOut = fopen(crashFile, "a");
    if(crOut) {
      fprintf(crOut, "%s\n", key);
      fflush(crOut);
      int fd = fileno(crOut);
      if(fd >= 0) fsync(fd);
      fclose(crOut);
    }
  }
}

void opengothic_log_write(const char* fmt, ...) {
  char buf[2048];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);

  std::lock_guard<std::mutex> lock(s_logMutex);
  logBoth(buf);
}

void opengothic_log_device_info(
    const char* deviceName,
    uint32_t apiVersion,
    uint32_t driverVersion,
    uint32_t vendorId,
    uint32_t deviceId,
    const char* const* enabledExts, size_t extCount,
    const char* featureInfo
) {
  if(!s_config.loaded) opengothic_debug_init();

  std::lock_guard<std::mutex> lock(s_logMutex);
  std::string s;
  s.reserve(4096);
  s += "\n============================================================\n";
  s += "[VULKAN DEVICE INITIALIZED]\n";
  char line[256];
  snprintf(line, sizeof(line), "  Device: %s (Vendor: 0x%04x, Device: 0x%04x)\n", deviceName, vendorId, deviceId);
  s += line;
  snprintf(line, sizeof(line), "  Vulkan API: %u.%u.%u, Driver: 0x%x (%u)\n",
           apiVersion >> 22, (apiVersion >> 12) & 0x3ff, apiVersion & 0xfff,
           driverVersion, driverVersion);
  s += line;
  s += "  Enabled Extensions (" + std::to_string(extCount) + "):\n";
  for(size_t i = 0; i < extCount; ++i) {
    s += "    - ";
    s += enabledExts[i];
    s += "\n";
  }
  if(featureInfo && featureInfo[0]) {
    s += "  Features & Limits:\n";
    s += featureInfo;
    s += "\n";
  }
  s += "============================================================\n";
  logBoth(s.c_str());
}

uint32_t opengothic_log_graphics_pipeline_begin(
    size_t stageCount,
    const char* const* stageNames,
    const char* const* shaderSources,
    const char* const* shaderDefines,
    const size_t* shaderSizes,
    const uint32_t* shaderHashes,
    const char* topology,
    const char* renderState,
    size_t colorAttachmentCount,
    const char* const* colorFormats,
    const char* depthFormat,
    const char* stencilFormat,
    uint32_t viewMask,
    size_t declSize,
    const char* const* declFormats,
    size_t stride,
    int isBindless,
    int isDynamicRendering,
    uint64_t renderPassHandle
) {
  if(!s_config.loaded) opengothic_debug_init();
  uint32_t id = s_pipelineSeq.fetch_add(1);

  if(!s_config.log_pipelines)
    return id;

  std::lock_guard<std::mutex> lock(s_logMutex);
  std::string s;
  s.reserve(2048);

  char hdr[256];
  snprintf(hdr, sizeof(hdr), "\n[PIPELINE #%u START] Graphics (Stages: %zu, Bindless: %d, DynRen: %d, RenderPass: 0x%lx)\n",
           id, stageCount, isBindless, isDynamicRendering, (unsigned long)renderPassHandle);
  s += hdr;

  for(size_t i = 0; i < stageCount; ++i) {
    char stageBuf[512];
    const char* defs = (shaderDefines && shaderDefines[i] && shaderDefines[i][0]) ? shaderDefines[i] : "none";
    snprintf(stageBuf, sizeof(stageBuf), "  Stage[%zu] %-8s: %s [defines: %s] (size=%zu, hash=0x%08x)\n",
             i, stageNames[i], shaderSources[i] ? shaderSources[i] : "(unknown)",
             defs, shaderSizes[i], shaderHashes[i]);
    s += stageBuf;
  }

  s += "  Topology: ";
  s += topology ? topology : "Triangles";
  s += "\n";

  if(renderState) {
    s += "  State: ";
    s += renderState;
    s += "\n";
  }

  char fboBuf[256];
  snprintf(fboBuf, sizeof(fboBuf), "  FBO: colorAttachments=%zu, depth=%s, stencil=%s, viewMask=0x%x\n",
           colorAttachmentCount, depthFormat ? depthFormat : "NONE",
           stencilFormat ? stencilFormat : "NONE", viewMask);
  s += fboBuf;

  for(size_t i = 0; i < colorAttachmentCount; ++i) {
    char attBuf[128];
    snprintf(attBuf, sizeof(attBuf), "    Color[%zu]: %s\n", i, colorFormats[i] ? colorFormats[i] : "UNKNOWN");
    s += attBuf;
  }

  char vboBuf[256];
  snprintf(vboBuf, sizeof(vboBuf), "  VBO: stride=%zu, attributes=%zu\n", stride, declSize);
  s += vboBuf;
  for(size_t i = 0; i < declSize; ++i) {
    char attrBuf[128];
    snprintf(attrBuf, sizeof(attrBuf), "    Attr[%zu]: %s\n", i, declFormats[i] ? declFormats[i] : "UNKNOWN");
    s += attrBuf;
  }

  s += "  >> Calling vkCreateGraphicsPipelines...\n";
  logBoth(s.c_str());
  return id;
}

void opengothic_log_graphics_pipeline_end(uint32_t seq, int status) {
  if(!s_config.log_pipelines)
    return;

  std::lock_guard<std::mutex> lock(s_logMutex);
  char buf[128];
  if(status == 0) {
    snprintf(buf, sizeof(buf), "[PIPELINE #%u OK]\n", seq);
  } else {
    snprintf(buf, sizeof(buf), "[PIPELINE #%u FAILED error=%d]\n", seq, status);
  }
  logBoth(buf);
}

uint32_t opengothic_log_compute_pipeline_begin(
    const char* shaderSource,
    size_t shaderSize,
    uint32_t shaderHash,
    int wgX, int wgY, int wgZ,
    int isUpdateAfterBind
) {
  if(!s_config.loaded) opengothic_debug_init();
  uint32_t id = s_pipelineSeq.fetch_add(1);

  if(!s_config.log_pipelines)
    return id;

  std::lock_guard<std::mutex> lock(s_logMutex);
  char buf[512];
  snprintf(buf, sizeof(buf),
    "\n[PIPELINE #%u START] Compute: %s (size=%zu, hash=0x%08x, wg=[%d,%d,%d], updateAfterBind=%d)\n"
    "  >> Calling vkCreateComputePipelines...\n",
    id, shaderSource ? shaderSource : "(unknown)", shaderSize, shaderHash,
    wgX, wgY, wgZ, isUpdateAfterBind
  );
  logBoth(buf);
  return id;
}

void opengothic_log_compute_pipeline_end(uint32_t seq, int status) {
  if(!s_config.log_pipelines)
    return;

  std::lock_guard<std::mutex> lock(s_logMutex);
  char buf[128];
  if(status == 0) {
    snprintf(buf, sizeof(buf), "[PIPELINE #%u OK]\n", seq);
  } else {
    snprintf(buf, sizeof(buf), "[PIPELINE #%u FAILED error=%d]\n", seq, status);
  }
  logBoth(buf);
}

} // extern "C"
