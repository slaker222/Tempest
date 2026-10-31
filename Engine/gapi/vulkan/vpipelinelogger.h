#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Debug Configuration API
// the folder of the game that runs: debug.cfg and the logs are kept there (default: the first Gothic2/Gothic folder found)
void opengothic_debug_set_game_dir(const char* dir);
void opengothic_debug_init();
void opengothic_debug_check_reload();
int  opengothic_debug_no_bindless();
int  opengothic_debug_no_dynrendering();
int  opengothic_debug_no_multidraw();
int  opengothic_debug_skip_hiz();
int  opengothic_debug_skip_shadows();
int  opengothic_debug_skip_lights();
int  opengothic_debug_skip_fog();
int  opengothic_debug_skip_gbuf();
int  opengothic_debug_skip_atest();
int  opengothic_debug_no_cmd_cull();
int  opengothic_debug_spirv_noopt();
int  opengothic_debug_no_atest_depth();
int  opengothic_debug_prewarm();
int  opengothic_debug_render_scale();
int  opengothic_debug_simple_render();
int  opengothic_debug_freeze_time();
int  opengothic_debug_draw_dist();
int  opengothic_debug_world_dist();
int  opengothic_debug_shadows();
int  opengothic_debug_ssao();
int  opengothic_debug_reflections();
int  opengothic_debug_show_fps();
int  opengothic_debug_fps_unlimited();
// 1 = use full-scene HiZ depth prepass (desktop default); Android default 0 (prepass costs more than occlusion saves on Adreno)
int  opengothic_debug_use_hiz_prepass();
int  opengothic_debug_force_hiz();

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
