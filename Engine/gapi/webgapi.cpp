#if defined(__EMSCRIPTEN__)

#include "webgapi.h"

#include <Tempest/Log>
#include <Tempest/Pixmap>
#include <Tempest/Except>
#include <Tempest/RenderState>

#include <GLES3/gl3.h>
#include <emscripten.h>
#include <emscripten/html5.h>

#include "gapi/shaderreflection.h"

#include "thirdparty/spirv_cross/spirv_glsl.hpp"

#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

using namespace Tempest;

// ============================================================================
// WebGL2 implementation of the Tempest graphics api.
//
// Shader modules arrive as SPIR-V (the engine embeds .sprv only) and are
// cross-compiled to GLSL ES 3.00 at pipeline creation time with spirv-cross.
// Pipelines using features beyond GLES3 (SSBO reads, compute, mesh shaders)
// fail to translate and are created as "unsupported" — their draws are
// skipped, which keeps the engine running while the 3D world rendering
// remains unimplemented (menus and UI are fully covered by this path).
// ============================================================================

namespace {

GLenum glFormat(TextureFormat f) {
  switch(f) {
    case TextureFormat::R8:        return GL_R8;
    case TextureFormat::RG8:       return GL_RG8;
    case TextureFormat::RGB8:      return GL_RGB8;
    case TextureFormat::RGBA8:     return GL_RGBA8;
    // GLES3 has no 16-bit normalized sized formats; fall back to 8-bit
    case TextureFormat::R16:       return GL_R8;
    case TextureFormat::RG16:      return GL_RG8;
    case TextureFormat::RGB16:     return GL_RGB8;
    case TextureFormat::RGBA16:    return GL_RGBA8;
    case TextureFormat::R32F:      return GL_R32F;
    case TextureFormat::RG32F:     return GL_RG32F;
    case TextureFormat::RGB32F:    return GL_RGB32F;
    case TextureFormat::RGBA32F:   return GL_RGBA32F;
    case TextureFormat::R32U:      return GL_R32UI;
    case TextureFormat::RG32U:     return GL_RG32UI;
    case TextureFormat::RGB32U:    return GL_RGB32UI;
    case TextureFormat::RGBA32U:   return GL_RGBA32UI;
    case TextureFormat::R11G11B10UF: return GL_R11F_G11F_B10F;
    case TextureFormat::Depth16:   return GL_DEPTH_COMPONENT16;
    case TextureFormat::Depth24x8: return GL_DEPTH_COMPONENT24;
    case TextureFormat::Depth32F:  return GL_DEPTH_COMPONENT32F;
    case TextureFormat::RGBA16F:   return GL_RGBA16F;
    default:                       return GL_RGBA8;
    }
  }

// `format` argument of glTexImage2D — must be a valid combination with the
// sized internal format and glType(), even for NULL-data allocations
GLenum glUploadFormat(TextureFormat f) {
  switch(f) {
    case TextureFormat::R8:
    case TextureFormat::R16:
    case TextureFormat::R32F:      return GL_RED;
    case TextureFormat::RG8:
    case TextureFormat::RG16:
    case TextureFormat::RG32F:     return GL_RG;
    case TextureFormat::R32U:
    case TextureFormat::RG32U:
    case TextureFormat::RGB32U:
    case TextureFormat::RGBA32U:   return GL_RED_INTEGER;
    case TextureFormat::R11G11B10UF: return GL_RGB;
    case TextureFormat::Depth16:
    case TextureFormat::Depth24x8:
    case TextureFormat::Depth32F:  return GL_DEPTH_COMPONENT;
    default:                       return GL_RGBA;
    }
  }

GLenum glType(TextureFormat f) {
  switch(f) {
    // 16-bit normalized formats are backed by 8-bit storage (see glFormat)
    case TextureFormat::R16:
    case TextureFormat::RG16:
    case TextureFormat::RGB16:
    case TextureFormat::RGBA16:   return GL_UNSIGNED_BYTE;
    case TextureFormat::R32F:
    case TextureFormat::RG32F:
    case TextureFormat::RGB32F:
    case TextureFormat::RGBA32F:
    case TextureFormat::R11G11B10UF: return GL_FLOAT;
    case TextureFormat::R32U:
    case TextureFormat::RG32U:
    case TextureFormat::RGB32U:
    case TextureFormat::RGBA32U:  return GL_UNSIGNED_INT;
    default:                      return GL_UNSIGNED_BYTE;
    }
  }

bool isDepthFrm(TextureFormat f) {
  return f==TextureFormat::Depth16 || f==TextureFormat::Depth24x8 ||
         f==TextureFormat::Depth24S8 || f==TextureFormat::Depth32F;
  }

GLenum glBlendFactor(RenderState::BlendMode m) {
  switch(m) {
    case RenderState::BlendMode::Zero:               return GL_ZERO;
    case RenderState::BlendMode::One:                return GL_ONE;
    case RenderState::BlendMode::SrcColor:           return GL_SRC_COLOR;
    case RenderState::BlendMode::OneMinusSrcColor:   return GL_ONE_MINUS_SRC_COLOR;
    case RenderState::BlendMode::SrcAlpha:           return GL_SRC_ALPHA;
    case RenderState::BlendMode::OneMinusSrcAlpha:   return GL_ONE_MINUS_SRC_ALPHA;
    case RenderState::BlendMode::DstAlpha:           return GL_DST_ALPHA;
    case RenderState::BlendMode::OneMinusDstAlpha:   return GL_ONE_MINUS_DST_ALPHA;
    case RenderState::BlendMode::DstColor:           return GL_DST_COLOR;
    case RenderState::BlendMode::OneMinusDstColor:   return GL_ONE_MINUS_DST_COLOR;
    case RenderState::BlendMode::SrcAlphaSaturate:   return GL_SRC_ALPHA_SATURATE;
    }
  return GL_ONE;
  }

GLenum glBlendEq(RenderState::BlendOp o) {
  switch(o) {
    case RenderState::BlendOp::Add:             return GL_FUNC_ADD;
    case RenderState::BlendOp::Subtract:        return GL_FUNC_SUBTRACT;
    case RenderState::BlendOp::ReverseSubtract: return GL_FUNC_REVERSE_SUBTRACT;
    case RenderState::BlendOp::Min:             return GL_MIN;
    case RenderState::BlendOp::Max:             return GL_MAX;
    }
  return GL_FUNC_ADD;
  }

GLenum glZTest(RenderState::ZTestMode z) {
  switch(z) {
    case RenderState::ZTestMode::Always:  return GL_ALWAYS;
    case RenderState::ZTestMode::Never:   return GL_NEVER;
    case RenderState::ZTestMode::Greater: return GL_GREATER;
    case RenderState::ZTestMode::Less:    return GL_LESS;
    case RenderState::ZTestMode::GEqual:  return GL_GEQUAL;
    case RenderState::ZTestMode::LEqual:  return GL_LEQUAL;
    case RenderState::ZTestMode::NoEqual: return GL_NOTEQUAL;
    case RenderState::ZTestMode::Equal:   return GL_EQUAL;
    }
  return GL_ALWAYS;
  }

GLenum glTopology(Topology t) {
  switch(t) {
    case Points:    return GL_POINTS;
    case Lines:     return GL_LINES;
    case Triangles: return GL_TRIANGLES;
    }
  return GL_TRIANGLES;
  }

GLenum glClamp(ClampMode c) {
  switch(c) {
    case ClampMode::ClampToBorder:  return GL_CLAMP_TO_EDGE;
    case ClampMode::ClampToEdge:    return GL_CLAMP_TO_EDGE;
    case ClampMode::MirroredRepeat: return GL_MIRRORED_REPEAT;
    case ClampMode::Repeat:         return GL_REPEAT;
    default:                        return GL_REPEAT;
    }
  }

// The GL context is created lazily on first use and shared by everything:
// Device builds the builtin pipelines (brush shader) long before the first
// swapchain exists, so the context must not be owned by the swapchain.
EMSCRIPTEN_WEBGL_CONTEXT_HANDLE g_glContext = 0;
GLuint g_pushUbo = 0; // global UBO backing emulated push constants (binding 0)

void ensureContext() {
  if(g_glContext > 0) {
    emscripten_webgl_make_context_current(g_glContext);
    return;
    }
  EmscriptenWebGLContextAttributes attrs;
  emscripten_webgl_init_context_attributes(&attrs);
  attrs.majorVersion = 2;
  attrs.minorVersion = 0;
  attrs.alpha = 0;
  attrs.depth = 1;
  attrs.stencil = 0;
  attrs.antialias = 1;
  // Keep the drawing buffer alive between frames. With 0 the compositor
  // invalidates it right after each composite, so any readback made at an
  // arbitrary moment (screenshot, drawImage, toDataURL) sees transparent
  // black even though the frame was presented correctly.
  attrs.preserveDrawingBuffer = 1;
  attrs.powerPreference = EM_WEBGL_POWER_PREFERENCE_HIGH_PERFORMANCE;

  g_glContext = emscripten_webgl_create_context("#canvas", &attrs);
  if(g_glContext > 0) {
    emscripten_webgl_make_context_current(g_glContext);
    Log::i("WebGL 2.0 context created successfully on #canvas");
    glEnable(GL_DEPTH_TEST);
    // shaders are Y-flipped against GL (see compileStage): a Vulkan-CCW
    // triangle becomes CW in GL window space
    glFrontFace(GL_CW);
    // emulated push constants: one global UBO on binding point 0
    glGenBuffers(1, &g_pushUbo);
    glBindBuffer(GL_UNIFORM_BUFFER, g_pushUbo);
    glBufferData(GL_UNIFORM_BUFFER, 1024, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, g_pushUbo);
    // needed to render into float attachments (R11F_G11F_B10F, R32F, ...)
    // and to filter them linearly
    EM_ASM({
      const c = (typeof Module !== 'undefined' && Module.canvas) ? Module.canvas : document.getElementById('canvas');
      const gl = c ? c.getContext('webgl2') : null;
      if(gl) {
        gl.getExtension('EXT_color_buffer_float');
        gl.getExtension('OES_texture_float_linear');
        }
      });
    } else {
    Log::e("Failed to create WebGL 2.0 context on #canvas");
    }
  }

uint32_t g_pipeOk    = 0;
uint32_t g_pipeFail  = 0;
bool     g_pipeDumped= false;

// per-diagnostic-window draw statistics (reset in present())
static int     g_dbg        = 1; // cached copy of window.__OG_GAPI_DEBUG
static uint64_t g_nDraws    = 0;
static uint64_t g_nIdxDraws = 0;
static uint64_t g_nTexBinds = 0;
static uint64_t g_nToScreen = 0;
static size_t  g_lastVerts  = 0;
static size_t  g_lastIdx    = 0;
static size_t  g_lastStride = 0;
static size_t  g_lastOffset = 0;
static GLuint  g_lastProg   = 0;

struct WebDevice : AbstractGraphicsApi::Device {
  WebDevice() {
    ensureContext();
    }
  void waitIdle() override {}
  };

struct WebBuffer : AbstractGraphicsApi::Buffer {
  std::vector<uint8_t> data;
  // GLES3 forbids client-side vertex pointers: data must live in a GL buffer
  GLuint               gpu    = 0;
  bool                 dirty  = true;

  WebBuffer(size_t sz, const void* mem) {
    data.resize(sz);
    if(mem != nullptr) {
      std::memcpy(data.data(), mem, sz);
      }
  }

  ~WebBuffer() override {
    if(gpu!=0)
      glDeleteBuffers(1, &gpu);
    }

  void update(const void* src, size_t off, size_t sz) override {
    if(off + sz <= data.size() && src != nullptr) {
      std::memcpy(data.data() + off, src, sz);
      dirty = true;
    }
  }

  void read(void* dst, size_t off, size_t sz) override {
    if(off + sz <= data.size() && dst != nullptr) {
      std::memcpy(dst, data.data() + off, sz);
    }
  }

  void ensureGpu(GLenum target) {
    if(gpu==0) {
      glGenBuffers(1, &gpu);
      dirty = true;
      }
    if(dirty) {
      glBindBuffer(target, gpu);
      glBufferData(target, data.size(), data.data(), GL_DYNAMIC_DRAW);
      dirty = false;
      }
    else {
      glBindBuffer(target, gpu);
      }
    }
  };

struct WebTexture : AbstractGraphicsApi::Texture {
  uint32_t      width = 0;
  uint32_t      height = 0;
  uint32_t      mips = 1;
  TextureFormat frm = TextureFormat::RGBA8;
  GLuint        tex = 0;
  GLuint        fbo = 0;
  // sampler state cache to avoid re-setting parameters on every draw
  bool          smpValid = false;
  Sampler       smp;

  uint32_t mipCount() const override { return mips; }

  void alloc() {
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    const GLenum upFmt = glUploadFormat(frm);
    glTexImage2D(GL_TEXTURE_2D, 0, glFormat(frm), width, height, 0,
                 upFmt, glType(frm), nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    if(frm==TextureFormat::R8 || frm==TextureFormat::R16 || frm==TextureFormat::R32F) {
      // GLES3 has no texture swizzle; emulate ComponentMapping at upload time by expanding to RGBA
      }
    }

  void upload(const Pixmap& p, uint32_t mip) {
    if(tex==0)
      glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    const uint8_t* src = static_cast<const uint8_t*>(p.data());
    std::vector<uint8_t> cvt;

    TextureFormat pfrm = p.format();
    if(pfrm==TextureFormat::R8) {
      // expand to RGBA (ComponentMapping rrrr for single-channel fonts etc.)
      cvt.resize(size_t(p.w())*p.h()*4);
      const size_t n = size_t(p.w())*p.h();
      for(size_t i=0;i<n;++i) {
        cvt[i*4+0] = src[i];
        cvt[i*4+1] = src[i];
        cvt[i*4+2] = src[i];
        cvt[i*4+3] = src[i];
        }
      glTexImage2D(GL_TEXTURE_2D, GLint(mip), GL_RGBA8, p.w(), p.h(), 0, GL_RGBA, GL_UNSIGNED_BYTE, cvt.data());
      }
    else if(pfrm==TextureFormat::RGBA8) {
      glTexImage2D(GL_TEXTURE_2D, GLint(mip), GL_RGBA8, p.w(), p.h(), 0, GL_RGBA, GL_UNSIGNED_BYTE, src);
      }
    else if(pfrm==TextureFormat::RGBA16F) {
      glTexImage2D(GL_TEXTURE_2D, GLint(mip), GL_RGBA16F, p.w(), p.h(), 0, GL_RGBA, GL_HALF_FLOAT, src);
      }
    else {
      glTexImage2D(GL_TEXTURE_2D, GLint(mip), glFormat(pfrm), p.w(), p.h(), 0, GL_RGBA, glType(pfrm), src);
      }
    }

  void ensureFbo() {
    if(fbo!=0 || tex==0 || isDepthFrm(frm))
      return;
    glGenFramebuffers(1, &fbo);
    GLint curFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &curFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, GLuint(curFbo));
    }

  void applySampler(const Sampler& s) {
    if(tex==0)
      return;
    if(smpValid &&
       s.minFilter==smp.minFilter && s.magFilter==smp.magFilter &&
       s.uClamp==smp.uClamp && s.vClamp==smp.vClamp)
      return;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    s.minFilter==Filter::Nearest ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    s.magFilter==Filter::Nearest ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, glClamp(s.uClamp));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, glClamp(s.vClamp));
    smp      = s;
    smpValid = true;
    }

  ~WebTexture() override {
    if(tex!=0)
      glDeleteTextures(1, &tex);
    if(fbo!=0)
      glDeleteFramebuffers(1, &fbo);
    }
  };

struct WebShader : AbstractGraphicsApi::Shader {
  std::vector<uint32_t> spirv;
  };

struct WebPipeline : AbstractGraphicsApi::Pipeline {
  GLuint                 prog = 0;
  bool                   ok = false;
  GLenum                 topology = GL_TRIANGLES;

  RenderState            st;

  struct Attrib {
    GLuint  location = 0;
    GLint   size = 0;
    size_t  offset = 0;
    };
  std::vector<Attrib>   attribs;

  struct SamplerBinding {
    size_t  slot = 0;
    GLint   uniformLocation = -1;
    GLuint  unit = 0;
    };
  std::vector<SamplerBinding> samplers;

  IVec3 workGroupSize() const override { return IVec3(1, 1, 1); }
  size_t sizeofBuffer(size_t, size_t) const override { return 0; }
  };

struct WebCompPipeline : AbstractGraphicsApi::CompPipeline {
  IVec3 workGroupSize() const override { return IVec3(1, 1, 1); }
  size_t sizeofBuffer(size_t, size_t) const override { return 0; }
  };

struct WebDescArray : AbstractGraphicsApi::DescArray {};

GLuint compileStage(spirv_cross::CompilerGLSL& comp, GLenum stageType) {  ensureContext();
  try {
    spirv_cross::CompilerGLSL::Options opts;
    opts.es = true;
    opts.version = 300;
    opts.enable_420pack_extension = false;
    // The engine's shaders are authored against the Vulkan clip space:
    // +Y points down and Z spans [0;1]. GL expects +Y up and Z in [-1;1].
    // These two options patch the vertex stage accordingly; with the Y flip
    // the window-space winding also inverts, so glFrontFace(GL_CW) is set
    // once at context creation.
    opts.vertex.fixup_clipspace = true;
    opts.vertex.flip_vert_y     = true;
    comp.set_common_options(opts);

    const std::string src = comp.compile();
    if(src.empty())
      return 0;

    const GLchar* csrc = src.c_str();
    GLuint stage = glCreateShader(stageType);
    glShaderSource(stage, 1, &csrc, nullptr);
    glCompileShader(stage);
    GLint ok = GL_FALSE;
    glGetShaderiv(stage, GL_COMPILE_STATUS, &ok);
    if(ok!=GL_TRUE) {
      GLchar log[2048];
      GLsizei len = 0;
      glGetShaderInfoLog(stage, sizeof(log)-1, &len, log);
      Log::e("[WebGapi] GLSL ES compile failed: ", std::string_view(log, size_t(len)));
      std::string_view sv = src;
      auto cut = std::min<size_t>(sv.size(), 900);
      Log::e("[WebGapi] shader source head: ", sv.substr(0, cut));
      glDeleteShader(stage);
      return 0;
      }
    return stage;
    }
  catch(const std::exception& e) {
    Log::e("[WebGapi] spirv-cross translation failed: ", e.what());
    return 0;
    }
  }

std::unique_ptr<WebPipeline> buildGraphicsPipeline(const RenderState& st, Topology tp,
                                                   const WebShader* vert, const WebShader* frag) {
  std::unique_ptr<WebPipeline> p(new WebPipeline());
  p->st        = st;
  p->topology  = glTopology(tp);

  ensureContext();

  GLuint vsh = 0, fsh = 0;
  try {
    spirv_cross::CompilerGLSL vc(vert->spirv);
    spirv_cross::CompilerGLSL fc(frag->spirv);

    if(vc.get_execution_model()!=spv::ExecutionModelVertex ||
       fc.get_execution_model()!=spv::ExecutionModelFragment)
      return p;

    // GLSL ES 3.00 links vertex/fragment varyings by NAME, while SPIR-V links
    // them by location. Give both sides deterministic names (`vout<N>`) keyed
    // by location so the pair always links.
    {
      auto vres = vc.get_shader_resources();
      auto fres = fc.get_shader_resources();
      size_t vIdx = 0;
      for(auto& vo : vres.stage_outputs) {
        uint32_t loc = vc.get_decoration(vo.id, spv::DecorationLocation);
        if(loc==uint32_t(-1))
          loc = uint32_t(vIdx);
        vc.set_name(vo.id, "vout" + std::to_string(loc));
        ++vIdx;
        }
      size_t fIdx = 0;
      for(auto& fi : fres.stage_inputs) {
        uint32_t loc = fc.get_decoration(fi.id, spv::DecorationLocation);
        if(loc==uint32_t(-1))
          loc = uint32_t(fIdx);
        fc.set_name(fi.id, "vout" + std::to_string(loc));
        ++fIdx;
        }
    }

    vsh = compileStage(vc, GL_VERTEX_SHADER);
    fsh = compileStage(fc, GL_FRAGMENT_SHADER);
    }
  catch(const std::exception& e) {
    Log::e("[WebGapi] pipeline reflection failed: ", e.what());
    return p;
    }

  struct ShaderGuard {
    GLuint v = 0, f = 0;
    ~ShaderGuard() {
      if(v!=0) glDeleteShader(v);
      if(f!=0) glDeleteShader(f);
      }
    } guard{vsh, fsh};

  if(vsh==0 || fsh==0)
    return p;

  p->prog = glCreateProgram();
  glAttachShader(p->prog, vsh);
  glAttachShader(p->prog, fsh);

  // bind vertex attribute locations as declared in SPIR-V, before linking
  size_t runningOffset = 0;
  try {
    spirv_cross::Compiler comp(vert->spirv);
    auto res = comp.get_shader_resources();
    struct AttrTemp { GLuint loc; GLint size; std::string name; };
    std::vector<AttrTemp> attrs;
    for(auto& r : res.stage_inputs) {
      AttrTemp a;
      a.loc  = comp.get_decoration(r.id, spv::DecorationLocation);
      auto& t = comp.get_type(r.type_id);
      a.size = GLint(t.vecsize!=0 ? t.vecsize : 1);
      a.name = r.name;
      attrs.push_back(a);
      }
    std::sort(attrs.begin(), attrs.end(), [](const AttrTemp& l, const AttrTemp& r){
      return l.loc < r.loc;
      });
    for(auto& a : attrs) {
      glBindAttribLocation(p->prog, a.loc, a.name.c_str());
      WebPipeline::Attrib at;
      at.location = a.loc;
      at.size     = a.size;
      at.offset   = runningOffset;
      runningOffset += sizeof(float)*size_t(a.size);
      p->attribs.push_back(at);
      }
    }
  catch(const std::exception&) {
    glDeleteProgram(p->prog);
    p->prog = 0;
    return p;
    }

  GLint ok = GL_FALSE;
  glLinkProgram(p->prog);
  glGetProgramiv(p->prog, GL_LINK_STATUS, &ok);
  if(ok!=GL_TRUE) {
    GLchar log[2048];
    GLsizei len = 0;
    glGetProgramInfoLog(p->prog, sizeof(log)-1, &len, log);
    Log::e("[WebGapi] program link failed: ", std::string_view(log, size_t(len)));
    glDeleteProgram(p->prog);
    p->prog = 0;
    return p;
    }

  // spirv-cross emits Vulkan push_constant blocks as plain UBOs in ES; bind
  // every such block to the global push UBO (uniform binding point 0)
  {
    GLint nBlocks = 0;
    glGetProgramiv(p->prog, GL_ACTIVE_UNIFORM_BLOCKS, &nBlocks);
    for(GLint i=0; i<nBlocks; ++i) {
      GLchar nm[128] = {};
      GLsizei len = 0;
      glGetActiveUniformBlockName(p->prog, GLuint(i), sizeof(nm)-1, &len, nm);
      std::string name(nm);
      if(name.find("push")!=std::string::npos || name.find("Push")!=std::string::npos)
        glUniformBlockBinding(p->prog, GLuint(i), 0);
      }
    }

  // map SPIR-V binding slots to GL sampler uniforms
  GLuint unit = 0;
  for(auto* sh : {vert, frag}) {
    if(sh==nullptr)
      continue;
    try {
      spirv_cross::Compiler comp(sh->spirv);
      auto res = comp.get_shader_resources();
      for(auto& r : res.sampled_images) {
        const auto slot = size_t(comp.get_decoration(r.id, spv::DecorationBinding));
        bool known = false;
        for(auto& s : p->samplers)
          if(s.slot==slot) { known = true; break; }
        if(known)
          continue;
        WebPipeline::SamplerBinding b;
        b.slot             = slot;
        b.unit             = unit;
        b.uniformLocation  = glGetUniformLocation(p->prog, r.name.c_str());
        if(b.uniformLocation>=0) {
          p->samplers.push_back(b);
          ++unit;
          }
        }
      }
    catch(const std::exception&) {
      }
    }

  p->ok = true;
  return p;
  }

struct WebCommandBuffer : AbstractGraphicsApi::CommandBuffer {
  bool              recording = false;
  WebPipeline*      curPipeline = nullptr;
  GLuint            curFbo = 0;

  // Last requested texture binding per slot. Callers bind either before or
  // after setPipeline (VectorImage binds first, world code binds after), so
  // bindings are stored here and (re-)applied by every setPipeline.
  struct Sticky {
    WebTexture* tex = nullptr;
    Sampler     smp;
    bool        valid = false;
    };
  Sticky sticky[16];

  void applyBindings(WebPipeline& p) {
    if(p.prog==0)
      return;
    for(auto& b : p.samplers) {
      if(b.slot>=16)
        continue;
      auto& s = sticky[b.slot];
      if(!s.valid || s.tex==nullptr)
        continue;
      glActiveTexture(GL_TEXTURE0 + b.unit);
      glBindTexture(GL_TEXTURE_2D, s.tex->tex);
      s.tex->applySampler(s.smp);
      glUniform1i(b.uniformLocation, GLint(b.unit));
      }
    }

  void disableAttribArrays() {
    for(size_t i=0;i<4;++i)
      glDisableVertexAttribArray(GLuint(i));
    }

  // Vulkan push constants are emulated through a global UBO (binding 0);
  // pipeline creation binds every push_constant block to it
  void setPushData(const void* data, size_t size) override {
    static bool traced = false;
    if(!traced) {
      traced = true;
      Log::i("[WebGapi] setPushData used (", size, " bytes)");
      EM_ASM({ try { console.trace('setPushData caller'); } catch(e) {} });
      }
    if(g_pushUbo!=0 && data!=nullptr && size<=1024) {
      glBindBuffer(GL_UNIFORM_BUFFER, g_pushUbo);
      glBufferSubData(GL_UNIFORM_BUFFER, 0, GLintptr(size), data);
      }
    }

  void applyPipelineState(WebPipeline& p) {
    if(p.prog==0)
      return;
    glUseProgram(p.prog);

    if(p.st.hasBlend()) {
      glEnable(GL_BLEND);
      glBlendFuncSeparate(glBlendFactor(p.st.blendSource()), glBlendFactor(p.st.blendDest()),
                          glBlendFactor(p.st.blendSource()), glBlendFactor(p.st.blendDest()));
      glBlendEquation(glBlendEq(p.st.blendOperation()));
      } else {
      glDisable(GL_BLEND);
      }

    const bool zt = p.st.zTestMode()!=RenderState::ZTestMode::Always;
    if(zt) {
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(glZTest(p.st.zTestMode()));
      } else {
      glDisable(GL_DEPTH_TEST);
      }
    glDepthMask(p.st.isZWriteEnabled() ? GL_TRUE : GL_FALSE);

    switch(p.st.cullFaceMode()) {
      case RenderState::CullMode::Back:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        break;
      case RenderState::CullMode::Front:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        break;
      case RenderState::CullMode::NoCull:
        glDisable(GL_CULL_FACE);
        break;
      }
    }

  void setupAttribs(WebPipeline& p, size_t stride) {
    if(stride==0)
      return;
    for(size_t i=0;i<4;++i)
      glDisableVertexAttribArray(GLuint(i));
    for(auto& a : p.attribs) {
      if(a.location>=4)
        continue;
      glEnableVertexAttribArray(a.location);
      glVertexAttribPointer(a.location, a.size, GL_FLOAT, GL_FALSE, GLint(stride),
                            reinterpret_cast<const void*>(a.offset));
      }
    }

  void beginRendering(const AttachmentDesc* desc, size_t descSize, uint32_t w, uint32_t h,
                      const TextureFormat* frm, AbstractGraphicsApi::Texture** tex,
                      AbstractGraphicsApi::Swapchain** swap, const uint32_t* mipLevel) override {
    (void)frm; (void)mipLevel; (void)tex;

    glDisable(GL_SCISSOR_TEST);

    bool toScreen = false;
    for(size_t i=0;i<descSize;++i) {
      if(swap!=nullptr && swap[i]!=nullptr)
        toScreen = true;
      }

    if(toScreen) {
      ++g_nToScreen;
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      curFbo = 0;
      if((g_dbg & 4)!=0) {
        // probe: red clear BEFORE the engine draws; survives -> no draws
        glClearColor(1.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        }
      } else {
      auto* t = (tex!=nullptr && tex[0]!=nullptr) ? static_cast<WebTexture*>(tex[0]) : nullptr;
      if(t!=nullptr) {
        t->ensureFbo();
        glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
        curFbo = t->fbo;
        } else {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        curFbo = 0;
        }
      }

    glViewport(0, 0, GLint(w), GLint(h));

    bool clrColor = false, clrDepth = false;
    Vec4 color(0.f);
    for(size_t i=0;i<descSize;++i) {
      if(desc[i].load!=AccessOp::Clear)
        continue;
      if(desc[i].zbuffer!=nullptr) {
        clrDepth = true;
        } else {
        clrColor = true;
        color = desc[i].clear;
        }
      }
    if(clrColor || clrDepth) {
      if(clrColor)
        glClearColor(color.x, color.y, color.z, color.w);
      else
        glClearColor(0.f, 0.f, 0.f, 1.f);
      glClear((clrColor ? GL_COLOR_BUFFER_BIT : 0u) |
              (clrDepth ? GL_DEPTH_BUFFER_BIT : 0u));
      }
    }
  void endRendering() override {}

  void barrier(const AbstractGraphicsApi::SyncDesc&, const AbstractGraphicsApi::BarrierDesc*, size_t) override {}

  void generateMipmap(AbstractGraphicsApi::Texture& t, uint32_t, uint32_t, uint32_t mipLevels) override {
    auto& wt = static_cast<WebTexture&>(t);
    if(wt.tex==0)
      return;
    wt.mips = mipLevels;
    glBindTexture(GL_TEXTURE_2D, wt.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
    }

  void copy(AbstractGraphicsApi::Buffer& src, size_t offset, AbstractGraphicsApi::Texture& dst,
            uint32_t width, uint32_t height, uint32_t mip) override {
    auto& s = static_cast<WebBuffer&>(src);
    auto& t = static_cast<WebTexture&>(dst);
    if(t.tex==0)
      t.alloc();
    if(s.data.size()<=offset)
      return;
    glBindTexture(GL_TEXTURE_2D, t.tex);
    const uint8_t* ptr = s.data.data()+offset;
    if(t.frm==TextureFormat::R8) {
      // single channel uploads are expanded to RGBA (GLES3 has no texture swizzle)
      std::vector<uint8_t> cvt(size_t(width)*height*4);
      for(size_t i=0;i<size_t(width)*height;++i) {
        cvt[i*4+0] = ptr[i];
        cvt[i*4+1] = ptr[i];
        cvt[i*4+2] = ptr[i];
        cvt[i*4+3] = ptr[i];
        }
      glTexSubImage2D(GL_TEXTURE_2D, GLint(mip), 0, 0, GLint(width), GLint(height), GL_RGBA, GL_UNSIGNED_BYTE, cvt.data());
      }
    else if(t.frm==TextureFormat::RGBA16F) {
      glTexSubImage2D(GL_TEXTURE_2D, GLint(mip), 0, 0, GLint(width), GLint(height), GL_RGBA, GL_HALF_FLOAT, ptr);
      }
    else {
      glTexSubImage2D(GL_TEXTURE_2D, GLint(mip), 0, 0, GLint(width), GLint(height), GL_RGBA, GL_UNSIGNED_BYTE, ptr);
      }
    }

  bool isRecording() const override { return recording; }
  void begin() override { recording = true; }
  void end()  override { recording = false; }
  void reset() override { recording = false; }

  void setPipeline(AbstractGraphicsApi::Pipeline& p) override {
    curPipeline = static_cast<WebPipeline*>(&p);
    applyPipelineState(*curPipeline);
    applyBindings(*curPipeline);
    }

  void setComputePipeline(AbstractGraphicsApi::CompPipeline&) override {
    curPipeline = nullptr;
    }

  void setBinding(size_t id, AbstractGraphicsApi::Texture* tex, uint32_t mip,
                  const ComponentMapping& m, const Sampler& smp) override {
    (void)mip; (void)m;
    if(id>=16)
      return;
    auto& s   = sticky[id];
    s.valid   = true;
    s.tex     = (tex!=nullptr) ? static_cast<WebTexture*>(tex) : nullptr;
    s.smp     = smp;
    if(curPipeline!=nullptr && curPipeline->prog!=0)
      applyBindings(*curPipeline);
    }

  void setBinding(size_t, AbstractGraphicsApi::Buffer*, size_t) override {}
  void setBinding(size_t, AbstractGraphicsApi::DescArray*) override {}
  void setBinding(size_t, AbstractGraphicsApi::AccelerationStructure*) override {}
  void setBinding(size_t, const Sampler&) override {}

  void setViewport(const Rect& r) override {
    glViewport(r.x, r.y, r.w, r.h);
    }
  void setScissor(const Rect& r) override {
    // Tempest passes GL-convention coordinates (same as setViewport, see
    // vkCmdSetScissor in the Vulkan backend) — no Y flip here
    if(r.w<=0 || r.h<=0) {
      glDisable(GL_SCISSOR_TEST);
      return;
      }
    glEnable(GL_SCISSOR_TEST);
    glScissor(r.x, r.y, r.w, r.h);
    }

  void draw(const AbstractGraphicsApi::Buffer* vbo, size_t stride, size_t offset,
            size_t vertexCount, size_t firstInstance, size_t instanceCount) override {
    // NOTE: `offset` is the first VERTEX index (see vkCmdDraw usage in the
    // Vulkan backend), not a byte offset
    (void)firstInstance;
    if(curPipeline==nullptr || !curPipeline->ok || vertexCount==0)
      return;
    ++g_nDraws;
    g_lastVerts  = vertexCount;
    g_lastStride = stride;
    g_lastOffset = offset;
    g_lastProg   = curPipeline->prog;
    if(vbo==nullptr) {
      // vertex-pull shader (e.g. bink.vert synthesizes a fullscreen quad from
      // gl_VertexID) — GLES3 forbids client pointers, and there is nothing to
      // upload; just draw with all attribute arrays disabled
      disableAttribArrays();
      glDrawArrays(curPipeline->topology, GLint(offset), GLint(vertexCount));
      return;
      }
    auto& b = *static_cast<WebBuffer*>(const_cast<AbstractGraphicsApi::Buffer*>(vbo));
    b.ensureGpu(GL_ARRAY_BUFFER);
    setupAttribs(*curPipeline, stride);
    if(instanceCount<=1)
      glDrawArrays(curPipeline->topology, GLint(offset), GLint(vertexCount));
    else
      glDrawArraysInstanced(curPipeline->topology, GLint(offset), GLint(vertexCount), GLint(instanceCount));
    }

  void drawIndexed(const AbstractGraphicsApi::Buffer* vbo, size_t stride, size_t voffset,
                   const AbstractGraphicsApi::Buffer& ibo, Detail::IndexClass cls, size_t ioffset,
                   size_t count, size_t firstInstance, size_t instanceCount) override {
    (void)voffset; (void)firstInstance;
    if(curPipeline==nullptr || !curPipeline->ok || vbo==nullptr || count==0)
      return;
    ++g_nIdxDraws;
    g_lastIdx    = count;
    g_lastStride = stride;
    g_lastProg   = curPipeline->prog;
    auto& b = *static_cast<WebBuffer*>(const_cast<AbstractGraphicsApi::Buffer*>(vbo));
    auto& i = const_cast<WebBuffer&>(static_cast<const WebBuffer&>(ibo));
    b.ensureGpu(GL_ARRAY_BUFFER);
    i.ensureGpu(GL_ELEMENT_ARRAY_BUFFER);
    setupAttribs(*curPipeline, stride);
    const GLenum idxType = (cls==Detail::IndexClass::i32) ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT;
    const size_t idxSize = (cls==Detail::IndexClass::i32) ? 4 : 2;
    if(ioffset*idxSize > i.data.size())
      return;
    const void* idxPtr = reinterpret_cast<const void*>(ioffset*idxSize);
    if(instanceCount<=1)
      glDrawElements(curPipeline->topology, GLint(count), idxType, idxPtr);
    else
      glDrawElementsInstanced(curPipeline->topology, GLint(count), idxType, idxPtr, GLint(instanceCount));
    }

  void drawIndirect(const AbstractGraphicsApi::Buffer&, size_t) override {}
  void dispatch(size_t, size_t, size_t) override {}
  void dispatchIndirect(const AbstractGraphicsApi::Buffer&, size_t) override {}
};

struct WebSwapchain : AbstractGraphicsApi::Swapchain {
  uint32_t width = 1280;
  uint32_t height = 720;

  void resizeToCanvas() {
    double cssW = 0, cssH = 0;
    if(emscripten_get_element_css_size("#canvas", &cssW, &cssH) >= 0 && cssW > 0 && cssH > 0) {
      width  = static_cast<uint32_t>(cssW);
      height = static_cast<uint32_t>(cssH);
      }
    double dpr = emscripten_get_device_pixel_ratio();
    const int pw = std::max(1, int(width  * dpr + 0.5));
    const int ph = std::max(1, int(height * dpr + 0.5));
    emscripten_set_canvas_element_size("#canvas", pw, ph);
  }

  WebSwapchain() {
    ensureContext();
    resizeToCanvas();
    Log::i("[WebGapi] swapchain created ", width, "x", height,
           " dpr=", emscripten_get_device_pixel_ratio());
    }

  ~WebSwapchain() override = default;

  void reset() override {
    ensureContext();
    resizeToCanvas();
    Log::i("[WebGapi] swapchain reset ", width, "x", height,
           " dpr=", emscripten_get_device_pixel_ratio());
    }

  uint32_t currentBackBufferIndex() override { return 0; }
  uint32_t imageCount() const override { return 2; }
  uint32_t w() const override { return width; }
  uint32_t h() const override { return height; }
  };

} // namespace

struct WebGapi::Impl {};

WebGapi::WebGapi(ApiFlags) {
  Log::i("WebGapi initialized for WebAssembly / Browser target");
  }

WebGapi::~WebGapi() = default;

std::vector<AbstractGraphicsApi::Props> WebGapi::devices() const {
  std::vector<Props> dev(1);
  dev[0].type = DeviceType::Virtual;
  std::snprintf(dev[0].name, sizeof(dev[0].name), "WebAssembly WebGL2 Engine");
  return dev;
  }

AbstractGraphicsApi::Device* WebGapi::createDevice(std::string_view) {
  return new WebDevice();
  }

AbstractGraphicsApi::Swapchain* WebGapi::createSwapchain(SystemApi::Window*, AbstractGraphicsApi::Device*) {
  return new WebSwapchain();
  }

AbstractGraphicsApi::PPipeline WebGapi::createPipeline(Device*, const RenderState& st, Topology tp,
                                                      const Shader*const* sh, size_t cnt) {
  const WebShader* vert = nullptr;
  const WebShader* frag = nullptr;
  for(size_t i=0;i<cnt;++i) {
    if(sh[i]==nullptr)
      continue;
    auto* s = static_cast<const WebShader*>(sh[i]);
    if(s->spirv.empty())
      continue;
    try {
      spirv_cross::Compiler c(s->spirv);
      if(c.get_execution_model()==spv::ExecutionModelVertex)
        vert = s;
      else if(c.get_execution_model()==spv::ExecutionModelFragment)
        frag = s;
      }
    catch(const std::exception&) {
      }
    }
  if(vert==nullptr || frag==nullptr) {
    static size_t failed = 0;
    if((failed++)%16==0)
      Log::e("[WebGapi] pipeline without vertex/fragment pair is not supported");
    return PPipeline(new WebPipeline());
    }
  auto p = buildGraphicsPipeline(st, tp, vert, frag);
  if(!p->ok) {
    ++g_pipeFail;
    if(!g_pipeDumped || (g_pipeFail % 32)==0) {
      g_pipeDumped = true;
      Log::e("[WebGapi] unable to build pipeline (unsupported shader?) failed=", g_pipeFail, " ok=", g_pipeOk);
      }
    }
  else {
    ++g_pipeOk;
    }
  return PPipeline(p.release());
  }

AbstractGraphicsApi::PCompPipeline WebGapi::createComputePipeline(Device*, Shader*) {
  return PCompPipeline(new WebCompPipeline());
  }

AbstractGraphicsApi::PShader WebGapi::createShader(Device*, const void* source, size_t src_size) {
  std::unique_ptr<WebShader> s(new WebShader());
  const size_t n = src_size/sizeof(uint32_t);
  s->spirv.assign(static_cast<const uint32_t*>(source), static_cast<const uint32_t*>(source)+n);
  return PShader(s.release());
  }

AbstractGraphicsApi::DescArray* WebGapi::createDescriptors(Device*, AbstractGraphicsApi::Texture**, size_t, uint32_t, const Sampler&) {
  return new WebDescArray();
  }

AbstractGraphicsApi::DescArray* WebGapi::createDescriptors(Device*, AbstractGraphicsApi::Texture**, size_t, uint32_t) {
  return new WebDescArray();
  }

AbstractGraphicsApi::DescArray* WebGapi::createDescriptors(Device*, AbstractGraphicsApi::Buffer**, size_t) {
  return new WebDescArray();
  }

AbstractGraphicsApi::PBuffer WebGapi::createBuffer(Device*, const void* mem, size_t size, MemUsage, BufferHeap) {
  return PBuffer(new WebBuffer(size, mem));
  }

AbstractGraphicsApi::PTexture WebGapi::createTexture(Device*, const Pixmap& p, TextureFormat frm, uint32_t mips) {
  std::unique_ptr<WebTexture> t(new WebTexture());
  t->width  = p.w();
  t->height = p.h();
  t->frm    = (p.format()==TextureFormat::Undefined) ? frm : p.format();
  t->mips   = std::max<uint32_t>(1, mips);
  t->upload(p, 0);
  return PTexture(t.release());
  }

AbstractGraphicsApi::PTexture WebGapi::createTexture(Device*, uint32_t w, uint32_t h, uint32_t mips, TextureFormat frm) {
  std::unique_ptr<WebTexture> t(new WebTexture());
  t->width  = w;
  t->height = h;
  t->frm    = frm;
  t->mips   = std::max<uint32_t>(1, mips);
  t->alloc();
  return PTexture(t.release());
  }

AbstractGraphicsApi::PTexture WebGapi::createStorage(Device* d, uint32_t w, uint32_t h, uint32_t mips, TextureFormat frm) {
  return createTexture(d, w, h, mips, frm);
  }

AbstractGraphicsApi::PTexture WebGapi::createStorage(Device* d, uint32_t w, uint32_t h, uint32_t, uint32_t mips, TextureFormat frm) {
  return createTexture(d, w, h, mips, frm);
  }

void WebGapi::readPixels(Device*, Pixmap& p, const PTexture tex, TextureFormat frm, uint32_t mip, uint32_t, uint32_t, bool) {
  (void)frm; (void)mip;
  auto* t = static_cast<WebTexture*>(tex.handler);
  if(t==nullptr || t->tex==0)
    return;
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, t->tex);
  // GLES3 cannot read back texture bytes directly; use a framebuffer blit
  GLuint fbo = 0;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->tex, 0);
  p = Pixmap(t->width, t->height, TextureFormat::RGBA8);
  glReadPixels(0, 0, GLint(t->width), GLint(t->height), GL_RGBA, GL_UNSIGNED_BYTE, p.data());
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  }

void WebGapi::readBytes(Device*, Buffer* buf, void* out, size_t size) {
  if(buf != nullptr && out != nullptr) {
    buf->read(out, 0, size);
    }
  }

AbstractGraphicsApi::CommandBuffer* WebGapi::createCommandBuffer(Device*) {
  return new WebCommandBuffer();
  }

void WebGapi::present(Device*, Swapchain*) {
  // Runtime diagnostics, switchable from the page console without a rebuild:
  //   window.__OG_GAPI_DEBUG = 1 -> periodic frame status log (default)
  //                              2 -> red clear probe (masks frame content!)
  //                              3 -> both
  static uint32_t frameNo = 0;
  ++frameNo;
  const int dbg = EM_ASM_INT({
    return (typeof window.__OG_GAPI_DEBUG === 'number') ? window.__OG_GAPI_DEBUG : 1;
    });
  g_dbg = dbg;

  if((dbg & 1)!=0 && ((frameNo % 120)==1)) {
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE cur = emscripten_webgl_get_current_context();
    GLint fbo = 0;
    GLint vp[4] = {};
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo);
    glGetIntegerv(GL_VIEWPORT, vp);
    const GLenum err = glGetError();
    const int cwh = EM_ASM_INT({
      const c = (typeof Module !== 'undefined' && Module.canvas) ? Module.canvas : document.getElementById('canvas');
      if(!c) return -1;
      return ((c.width & 0xFFFF) << 16) | Math.min(0xFFFF, c.height);
      });
    Log::i("[WebGapi] frame ", frameNo, ": fbo=", fbo, " vp=", vp[0], ",", vp[1], " ",
           vp[2], "x", vp[3], " canvas=", (cwh >> 16), "x", (cwh & 0xFFFF),
           " glErr=", int(err), " curCtx=", uint64_t(cur), " gCtx=", uint64_t(g_glContext),
           " | draws=", g_nDraws, " idx=", g_nIdxDraws, " binds=", g_nTexBinds,
           " toScr=", g_nToScreen, " lastV=", g_lastVerts, " lastI=", g_lastIdx,
           " lastStride=", g_lastStride, " lastOff=", g_lastOffset, " lastProg=", g_lastProg);
    if(cur != g_glContext)
      Log::e("[WebGapi] PRESENT ON WRONG CONTEXT: curCtx != gCtx");
    g_nDraws = 0; g_nIdxDraws = 0; g_nTexBinds = 0; g_nToScreen = 0;
    }

  if((dbg & 2)!=0) {
    glClearColor(1.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    }

  glFlush();
  }

struct WebFence : AbstractGraphicsApi::Fence {
  void wait() override {}
  bool wait(uint64_t) override { return true; }
  };

std::shared_ptr<AbstractGraphicsApi::Fence> WebGapi::submit(Device*, CommandBuffer*) {
  return std::make_shared<WebFence>();
  }

void WebGapi::getCaps(Device*, Props& props) {
  props.type = DeviceType::Virtual;
  std::snprintf(props.name, sizeof(props.name), "WebAssembly WebGL2 Engine");

  const uint64_t smpBits  = (uint64_t(1)<<uint64_t(TextureFormat::R8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RG8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGB8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R16)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RG16)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R32F)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RG32F)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGB32F)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA32F)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RG32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGB32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R11G11B10UF)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA16F));
  const uint64_t attBits  = (uint64_t(1)<<uint64_t(TextureFormat::R8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RG8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGB8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R16)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R32F)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R11G11B10UF)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA16F)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA32F));
  const uint64_t depthBits= (uint64_t(1)<<uint64_t(TextureFormat::Depth16)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::Depth24x8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::Depth32F));
  // storage images are declared as supported so resource creation succeeds;
  // GLES3 itself has no image load/store — compute dispatches are no-ops.
  // NOTE: R11G11B10UF stays OUT of storage bits on purpose — the engine
  // falls back to a reduced (non-compute) renderer path when it is absent
  const uint64_t storBits = (uint64_t(1)<<uint64_t(TextureFormat::R8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R16)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RG32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGB32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA32U)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::R32F)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA32F)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA8)) |
                            (uint64_t(1)<<uint64_t(TextureFormat::RGBA16F));
  props.setSamplerFormats(smpBits);
  props.setAttachFormats(attBits);
  props.setDepthFormats(depthBits);
  props.setStorageFormats(storBits);
  props.setAtomicFormats(0);
  }

#endif // __EMSCRIPTEN__
