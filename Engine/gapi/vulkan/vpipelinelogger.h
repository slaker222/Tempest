#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Debug Configuration API
void opengothic_debug_init();
int  opengothic_debug_no_bindless();
int  opengothic_debug_no_dynrendering();
int  opengothic_debug_no_multidraw();
int  opengothic_debug_skip_hiz();
int  opengothic_debug_skip_shadows();
int  opengothic_debug_skip_lights();
int  opengothic_debug_skip_fog();
int  opengothic_debug_skip_gbuf();
int  opengothic_debug_skip_atest();
int  opengothic_debug_spirv_noopt();
int  opengothic_debug_no_atest_depth();
int  opengothic_debug_prewarm();

// Prewarm Checkpoint API
int  opengothic_prewarm_check_crashed(const char* key);
void opengothic_prewarm_begin_attempt(const char* key);
void opengothic_prewarm_end_attempt(const char* key, int ok);

// Pipeline and Device Logger API
void opengothic_log_device_info(
    const char* deviceName,
    uint32_t apiVersion,
    uint32_t driverVersion,
    uint32_t vendorId,
    uint32_t deviceId,
    const char* const* enabledExts, size_t extCount,
    const char* featureInfo
);

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
);

void opengothic_log_graphics_pipeline_end(uint32_t seq, int status);

uint32_t opengothic_log_compute_pipeline_begin(
    const char* shaderSource,
    size_t shaderSize,
    uint32_t shaderHash,
    int wgX, int wgY, int wgZ,
    int isUpdateAfterBind
);

void opengothic_log_compute_pipeline_end(uint32_t seq, int status);

void opengothic_log_write(const char* fmt, ...);

#ifdef __cplusplus
}
#endif
