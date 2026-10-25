#if defined(__EMSCRIPTEN__)

#include "webgapi.h"

#include <Tempest/Log>
#include <Tempest/Pixmap>
#include <Tempest/Except>
#include <Tempest/RenderState>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#include <webgpu/webgpu.h>

#include "gapi/shader.h"
#include "gapi/shaderreflection.h"
#include "thirdparty/spirv_cross/spirv_cross.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using namespace Tempest;
using namespace Tempest::Detail;

// ============================================================================
// WebGPU implementation of the Tempest graphics api.
//
// Shader modules arrive as blobs produced by scripts/spv_web.py:
//   'WGSL' u32 spirvSize spirv[spirvSize] wgsl-text
// SPIR-V is used for reflection only, WGSL creates the GPU module. The blob
// rewrites push constants to a uniform at PushBinding and splits combined
// samplers: texture stays at N, its sampler moves to N + SamplerShift.
// ============================================================================

namespace {

constexpr uint32_t PushBinding  = 64;
constexpr uint32_t SamplerShift = 32;
constexpr uint32_t PushAlign    = 256;
constexpr size_t   PushChunk    = 64*1024;

struct Ctx {
  WGPUDevice        dev   = nullptr;
  WGPUQueue         queue = nullptr;
  WGPUSupportedLimits limits = {};
  bool              bc            = false;
  bool              f32filter     = false;
  bool              rg11b10render = false;
  bool              firstInstance = false;
  WGPUTextureFormat canvasFormat  = WGPUTextureFormat_BGRA8Unorm;
  uint64_t          uid           = 1;

  std::unordered_map<uint32_t, WGPUSampler> samplers;

  // deferred destruction: objects may still be referenced by command buffers
  // that are recorded but not submitted yet
  std::vector<WGPUTexture> deadTex;
  std::vector<WGPUBuffer>  deadBuf;

  // mip generation
  WGPUShaderModule  blitModule = nullptr;
  WGPUBindGroupLayout blitBgl  = nullptr;
  WGPUPipelineLayout  blitLay  = nullptr;
  std::unordered_map<uint32_t, WGPURenderPipeline> blitPso;

  uint32_t          errors = 0;

  struct Stats {
    uint32_t passes = 0, screenPasses = 0, draws = 0, skipped = 0, dispatches = 0, submits = 0;
    } st;
  uint32_t          frame = 0;
  int               dbg   = 0; // window.__OG_GAPI_DEBUG: 2 = magenta clear of screen passes
  };
Ctx g;

void onDeviceError(WGPUErrorType type, const char* msg, void*) {
  ++g.errors;
  if(g.errors<64)
    Log::e("[WebGPU] error(", int(type), "): ", msg ? msg : "");
  else if(g.errors==64)
    Log::e("[WebGPU] too many errors, further messages suppressed");
  }

// validation errors are reported asynchronously: tag them with the object name
void scopeBegin() {
  wgpuDevicePushErrorScope(g.dev, WGPUErrorFilter_Validation);
  }

void scopeEnd(std::string what) {
  auto* tag = new std::string(std::move(what));
  wgpuDevicePopErrorScope(g.dev, [](WGPUErrorType type, const char* msg, void* user) {
    auto* tag = static_cast<std::string*>(user);
    if(type!=WGPUErrorType_NoError) {
      ++g.errors;
      if(g.errors<200)
        Log::e("[WebGPU] ", *tag, ": ", msg ? msg : "");
      }
    delete tag;
    }, tag);
  }

uint32_t alignUp(uint32_t v, uint32_t a) { return (v+a-1)/a*a; }
uint64_t alignUp(uint64_t v, uint64_t a) { return (v+a-1)/a*a; }

struct FormatInfo {
  WGPUTextureFormat wgpu       = WGPUTextureFormat_Undefined;
  uint32_t          bpp        = 4;    // bytes per pixel; block bytes for compressed
  bool              renderable = false;
  bool              storage    = false;
  bool              depth      = false;
  bool              compressed = false;
  bool              integer    = false;
  bool              float32    = false;
  };

FormatInfo formatInfo(TextureFormat f) {
  FormatInfo r;
  switch(f) {
    case TextureFormat::R8:       r = {WGPUTextureFormat_R8Unorm,      1, true,  false}; break;
    case TextureFormat::RG8:      r = {WGPUTextureFormat_RG8Unorm,     2, true,  false}; break;
    case TextureFormat::RGBA8:    r = {WGPUTextureFormat_RGBA8Unorm,   4, true,  true }; break;
    // 16-bit normalized formats are not in core WebGPU: emulated with float storage
    case TextureFormat::R16:      r = {WGPUTextureFormat_R32Float,     4, true,  true }; r.float32 = true; break;
    case TextureFormat::RG16:     r = {WGPUTextureFormat_RG32Float,    8, true,  true }; r.float32 = true; break;
    case TextureFormat::RGBA16:   r = {WGPUTextureFormat_RGBA16Float,  8, true,  true }; break;
    case TextureFormat::R32F:     r = {WGPUTextureFormat_R32Float,     4, true,  true }; r.float32 = true; break;
    case TextureFormat::RG32F:    r = {WGPUTextureFormat_RG32Float,    8, true,  true }; r.float32 = true; break;
    case TextureFormat::RGBA32F:  r = {WGPUTextureFormat_RGBA32Float, 16, true,  true }; r.float32 = true; break;
    case TextureFormat::R32U:     r = {WGPUTextureFormat_R32Uint,      4, true,  true }; r.integer = true; break;
    case TextureFormat::RG32U:    r = {WGPUTextureFormat_RG32Uint,     8, true,  true }; r.integer = true; break;
    case TextureFormat::RGBA32U:  r = {WGPUTextureFormat_RGBA32Uint,  16, true,  true }; r.integer = true; break;
    case TextureFormat::RGBA16F:  r = {WGPUTextureFormat_RGBA16Float,  8, true,  true }; break;
    case TextureFormat::R11G11B10UF: r = {WGPUTextureFormat_RG11B10Ufloat, 4, g.rg11b10render, false}; break;
    case TextureFormat::Depth16:  r = {WGPUTextureFormat_Depth16Unorm, 2, true, false}; r.depth = true; break;
    case TextureFormat::Depth24x8:r = {WGPUTextureFormat_Depth24Plus,  4, true, false}; r.depth = true; break;
    case TextureFormat::Depth24S8:r = {WGPUTextureFormat_Depth24PlusStencil8, 4, true, false}; r.depth = true; break;
    case TextureFormat::Depth32F: r = {WGPUTextureFormat_Depth32Float, 4, true, false}; r.depth = true; break;
    case TextureFormat::DXT1:     r = {WGPUTextureFormat_BC1RGBAUnorm, 8,  false, false}; r.compressed = true; break;
    case TextureFormat::DXT3:     r = {WGPUTextureFormat_BC2RGBAUnorm, 16, false, false}; r.compressed = true; break;
    case TextureFormat::DXT5:     r = {WGPUTextureFormat_BC3RGBAUnorm, 16, false, false}; r.compressed = true; break;
    default: break;
    }
  return r;
  }

WGPUVertexFormat vertexFormat(Decl::ComponentType t) {
  switch(t) {
    case Decl::float1: return WGPUVertexFormat_Float32;
    case Decl::float2: return WGPUVertexFormat_Float32x2;
    case Decl::float3: return WGPUVertexFormat_Float32x3;
    case Decl::float4: return WGPUVertexFormat_Float32x4;
    case Decl::int1:   return WGPUVertexFormat_Sint32;
    case Decl::int2:   return WGPUVertexFormat_Sint32x2;
    case Decl::int3:   return WGPUVertexFormat_Sint32x3;
    case Decl::int4:   return WGPUVertexFormat_Sint32x4;
    case Decl::uint1:  return WGPUVertexFormat_Uint32;
    case Decl::uint2:  return WGPUVertexFormat_Uint32x2;
    case Decl::uint3:  return WGPUVertexFormat_Uint32x3;
    case Decl::uint4:  return WGPUVertexFormat_Uint32x4;
    default:           return WGPUVertexFormat_Float32;
    }
  }

WGPUBlendFactor blendFactor(RenderState::BlendMode m) {
  switch(m) {
    case RenderState::BlendMode::Zero:             return WGPUBlendFactor_Zero;
    case RenderState::BlendMode::One:              return WGPUBlendFactor_One;
    case RenderState::BlendMode::SrcColor:         return WGPUBlendFactor_Src;
    case RenderState::BlendMode::OneMinusSrcColor: return WGPUBlendFactor_OneMinusSrc;
    case RenderState::BlendMode::SrcAlpha:         return WGPUBlendFactor_SrcAlpha;
    case RenderState::BlendMode::OneMinusSrcAlpha: return WGPUBlendFactor_OneMinusSrcAlpha;
    case RenderState::BlendMode::DstAlpha:         return WGPUBlendFactor_DstAlpha;
    case RenderState::BlendMode::OneMinusDstAlpha: return WGPUBlendFactor_OneMinusDstAlpha;
    case RenderState::BlendMode::DstColor:         return WGPUBlendFactor_Dst;
    case RenderState::BlendMode::OneMinusDstColor: return WGPUBlendFactor_OneMinusDst;
    case RenderState::BlendMode::SrcAlphaSaturate: return WGPUBlendFactor_SrcAlphaSaturated;
    }
  return WGPUBlendFactor_One;
  }

WGPUBlendOperation blendOp(RenderState::BlendOp o) {
  switch(o) {
    case RenderState::BlendOp::Add:             return WGPUBlendOperation_Add;
    case RenderState::BlendOp::Subtract:        return WGPUBlendOperation_Subtract;
    case RenderState::BlendOp::ReverseSubtract: return WGPUBlendOperation_ReverseSubtract;
    case RenderState::BlendOp::Min:             return WGPUBlendOperation_Min;
    case RenderState::BlendOp::Max:             return WGPUBlendOperation_Max;
    }
  return WGPUBlendOperation_Add;
  }

WGPUCompareFunction compareFn(RenderState::ZTestMode z) {
  switch(z) {
    case RenderState::ZTestMode::Always:  return WGPUCompareFunction_Always;
    case RenderState::ZTestMode::Never:   return WGPUCompareFunction_Never;
    case RenderState::ZTestMode::Greater: return WGPUCompareFunction_Greater;
    case RenderState::ZTestMode::Less:    return WGPUCompareFunction_Less;
    case RenderState::ZTestMode::GEqual:  return WGPUCompareFunction_GreaterEqual;
    case RenderState::ZTestMode::LEqual:  return WGPUCompareFunction_LessEqual;
    case RenderState::ZTestMode::NoEqual: return WGPUCompareFunction_NotEqual;
    case RenderState::ZTestMode::Equal:   return WGPUCompareFunction_Equal;
    }
  return WGPUCompareFunction_Always;
  }

WGPUAddressMode addressMode(ClampMode c) {
  switch(c) {
    case ClampMode::ClampToBorder:
    case ClampMode::ClampToEdge:    return WGPUAddressMode_ClampToEdge;
    case ClampMode::MirroredRepeat: return WGPUAddressMode_MirrorRepeat;
    case ClampMode::Repeat:         return WGPUAddressMode_Repeat;
    default:                        return WGPUAddressMode_Repeat;
    }
  }

WGPUSampler sampler(const Sampler& s, bool nonFiltering, bool comparison) {
  Filter minF = s.minFilter, magF = s.magFilter, mipF = s.mipFilter;
  if(nonFiltering) {
    minF = magF = mipF = Filter::Nearest;
    }
  const bool aniso = s.anisotropic && !nonFiltering && !comparison &&
                     minF==Filter::Linear && magF==Filter::Linear && mipF==Filter::Linear;
  uint32_t key = uint32_t(minF) | uint32_t(magF)<<2 | uint32_t(mipF)<<4 |
                 uint32_t(s.uClamp)<<6 | uint32_t(s.vClamp)<<9 | uint32_t(s.wClamp)<<12 |
                 uint32_t(aniso)<<15 | uint32_t(comparison)<<16;
  auto it = g.samplers.find(key);
  if(it!=g.samplers.end())
    return it->second;

  WGPUSamplerDescriptor d = {};
  d.addressModeU  = addressMode(s.uClamp);
  d.addressModeV  = addressMode(s.vClamp);
  d.addressModeW  = addressMode(s.wClamp);
  d.magFilter     = magF==Filter::Linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
  d.minFilter     = minF==Filter::Linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
  d.mipmapFilter  = mipF==Filter::Linear ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
  d.lodMinClamp   = 0.f;
  d.lodMaxClamp   = 32.f;
  d.compare       = comparison ? WGPUCompareFunction_LessEqual : WGPUCompareFunction_Undefined;
  d.maxAnisotropy = aniso ? 16 : 1;
  auto smp = wgpuDeviceCreateSampler(g.dev, &d);
  g.samplers[key] = smp;
  return smp;
  }

// ---------------------------------------------------------------- resources

struct WDevice : AbstractGraphicsApi::Device {
  void waitIdle() override {}
  };

struct WBuffer : AbstractGraphicsApi::Buffer {
  WGPUBuffer           impl  = nullptr;
  uint64_t             size  = 0;  // requested size
  uint64_t             alloc = 0;  // aligned allocation size
  uint64_t             uid   = 0;
  std::vector<uint8_t> shadow;     // CPU copy of upload-heap buffers

  WBuffer(const void* mem, size_t sz, BufferHeap heap) {
    uid   = g.uid++;
    size  = sz;
    alloc = std::max<uint64_t>(16, alignUp(uint64_t(sz), uint64_t(4)));
    WGPUBufferDescriptor d = {};
    d.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Index | WGPUBufferUsage_Uniform |
              WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect |
              WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    d.size  = alloc;
    d.mappedAtCreation = false;
    impl = wgpuDeviceCreateBuffer(g.dev, &d);
    if(heap!=BufferHeap::Device)
      shadow.resize(size_t(alloc), 0);
    if(mem!=nullptr && sz>0)
      update(mem, 0, sz);
    }

  ~WBuffer() override {
    if(impl!=nullptr)
      g.deadBuf.push_back(impl);
    }

  void update(const void* data, size_t off, size_t sz) override {
    if(data==nullptr || sz==0 || off+sz>size)
      return;
    if(!shadow.empty())
      std::memcpy(shadow.data()+off, data, sz);
    if(off%4==0 && sz%4==0) {
      wgpuQueueWriteBuffer(g.queue, impl, off, data, sz);
      return;
      }
    // writeBuffer needs 4-byte granularity: widen the range
    const size_t b = off/4*4;
    const size_t e = std::min<size_t>(size_t(alloc), alignUp(uint64_t(off+sz), uint64_t(4)));
    std::vector<uint8_t> tmp(e-b, 0);
    if(!shadow.empty())
      std::memcpy(tmp.data(), shadow.data()+b, e-b);
    std::memcpy(tmp.data()+(off-b), data, sz);
    wgpuQueueWriteBuffer(g.queue, impl, b, tmp.data(), tmp.size());
    }

  void read(void* data, size_t off, size_t sz) override {
    if(data==nullptr || off+sz>size)
      return;
    if(!shadow.empty())
      std::memcpy(data, shadow.data()+off, sz); else
      std::memset(data, 0, sz);
    }
  };

struct WTexture : AbstractGraphicsApi::Texture {
  WGPUTexture       impl   = nullptr;
  TextureFormat     frm    = TextureFormat::Undefined;
  FormatInfo        info;
  uint32_t          w = 1, h = 1, d = 1;
  uint32_t          mips   = 1;
  bool              is3D   = false;
  uint64_t          uid    = 0;
  // key: mip (0xFFFF = all) | dim<<16
  std::unordered_map<uint32_t, WGPUTextureView> views;

  uint32_t mipCount() const override { return mips; }

  void create(WGPUTextureUsageFlags usage) {
    uid = g.uid++;
    WGPUTextureDescriptor td = {};
    td.usage         = usage;
    td.dimension     = is3D ? WGPUTextureDimension_3D : WGPUTextureDimension_2D;
    td.size          = {w, h, d};
    td.format        = info.wgpu;
    td.mipLevelCount = mips;
    td.sampleCount   = 1;
    impl = wgpuDeviceCreateTexture(g.dev, &td);
    }

  ~WTexture() override {
    for(auto& v:views)
      wgpuTextureViewRelease(v.second);
    if(impl!=nullptr)
      g.deadTex.push_back(impl);
    }

  WGPUTextureView view(uint32_t mip, WGPUTextureViewDimension dim) {
    if(dim==WGPUTextureViewDimension_Undefined)
      dim = is3D ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D;
    const uint32_t key = (mip==uint32_t(-1) ? 0xFFFFu : (mip & 0xFFFFu)) | (uint32_t(dim)<<16);
    auto it = views.find(key);
    if(it!=views.end())
      return it->second;
    WGPUTextureViewDescriptor vd = {};
    vd.format          = info.wgpu;
    vd.dimension       = dim;
    vd.baseMipLevel    = (mip==uint32_t(-1)) ? 0 : mip;
    vd.mipLevelCount   = (mip==uint32_t(-1)) ? mips : 1;
    vd.baseArrayLayer  = 0;
    vd.arrayLayerCount = 1;
    vd.aspect          = info.depth ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
    auto v = wgpuTextureCreateView(impl, &vd);
    views[key] = v;
    return v;
    }

  // render-attachment view: depth-stencil formats need the full aspect
  WGPUTextureView attachmentView(uint32_t mip) {
    const uint32_t key = (mip & 0xFFFFu) | (0xFFu<<16);
    auto it = views.find(key);
    if(it!=views.end())
      return it->second;
    WGPUTextureViewDescriptor vd = {};
    vd.format          = info.wgpu;
    vd.dimension       = WGPUTextureViewDimension_2D;
    vd.baseMipLevel    = mip;
    vd.mipLevelCount   = 1;
    vd.baseArrayLayer  = 0;
    vd.arrayLayerCount = 1;
    vd.aspect          = WGPUTextureAspect_All;
    auto v = wgpuTextureCreateView(impl, &vd);
    views[key] = v;
    return v;
    }
  };

struct WDescArray : AbstractGraphicsApi::DescArray {};

// ------------------------------------------------------------------ shaders

struct ResInfo {
  WGPUTextureViewDimension dim        = WGPUTextureViewDimension_2D;
  WGPUTextureSampleType    sampleType = WGPUTextureSampleType_Float;
  bool                     depth      = false; // shadow sampler (Depth=1 image)
  WGPUStorageTextureAccess access     = WGPUStorageTextureAccess_ReadWrite;
  WGPUTextureFormat        storageFmt = WGPUTextureFormat_Undefined;
  };

WGPUTextureFormat storageFormat(spv::ImageFormat f) {
  switch(f) {
    case spv::ImageFormatRgba32f:      return WGPUTextureFormat_RGBA32Float;
    case spv::ImageFormatRgba16f:      return WGPUTextureFormat_RGBA16Float;
    case spv::ImageFormatR32f:         return WGPUTextureFormat_R32Float;
    case spv::ImageFormatRgba8:        return WGPUTextureFormat_RGBA8Unorm;
    case spv::ImageFormatRg32f:        return WGPUTextureFormat_RG32Float;
    case spv::ImageFormatR16:          return WGPUTextureFormat_R32Float;
    case spv::ImageFormatRgba32ui:     return WGPUTextureFormat_RGBA32Uint;
    case spv::ImageFormatRg32ui:       return WGPUTextureFormat_RG32Uint;
    case spv::ImageFormatR32ui:        return WGPUTextureFormat_R32Uint;
    case spv::ImageFormatR32i:         return WGPUTextureFormat_R32Sint;
    case spv::ImageFormatR11fG11fB10f: return WGPUTextureFormat_RG11B10Ufloat;
    case spv::ImageFormatR8:           return WGPUTextureFormat_R8Unorm;
    default:                           return WGPUTextureFormat_Undefined;
    }
  }

struct WShader : Detail::Shader {
  WGPUShaderModule module = nullptr;
  std::string      name;
  ResInfo          res[MaxBindings];
  uint32_t         fragOutputs = 0; // bitmask of fragment output locations
  bool             fragUintOut[MaxFramebufferAttachments] = {};

  WShader(const uint32_t* spirv, size_t words, const char* wgsl, size_t wgslLen, std::string nm)
    : Detail::Shader(spirv, words*4) {
    name = nm.empty() ? std::string(dbgShortName()) : std::move(nm);
    reflect(spirv, words);
    if(wgsl!=nullptr && wgslLen>0) {
      std::string code(wgsl, wgslLen);
      WGPUShaderModuleWGSLDescriptor wd = {};
      wd.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
      wd.code        = code.c_str();
      WGPUShaderModuleDescriptor md = {};
      md.nextInChain = &wd.chain;
      md.label       = name.c_str();
      scopeBegin();
      module = wgpuDeviceCreateShaderModule(g.dev, &md);
      scopeEnd("shader " + name);
      }
    }

  ~WShader() override {
    if(module!=nullptr)
      wgpuShaderModuleRelease(module);
    }

  void reflect(const uint32_t* spirv, size_t words) {
    spirv_cross::Compiler comp(spirv, words);
    for(auto& b:lay) {
      if(b.layout>=MaxBindings)
        continue;
      auto& r = res[b.layout];
      if(b.cls==ShaderReflection::Texture || b.cls==ShaderReflection::Image ||
         b.cls==ShaderReflection::ImgR    || b.cls==ShaderReflection::ImgRW) {
        auto& t = comp.get_type_from_variable(b.spvId);
        switch(t.image.dim) {
          case spv::Dim3D:   r.dim = WGPUTextureViewDimension_3D;   break;
          case spv::DimCube: r.dim = WGPUTextureViewDimension_Cube; break;
          default:           r.dim = t.image.arrayed ? WGPUTextureViewDimension_2DArray : WGPUTextureViewDimension_2D; break;
          }
        auto& st = comp.get_type(t.image.type);
        if(st.basetype==spirv_cross::SPIRType::UInt)
          r.sampleType = WGPUTextureSampleType_Uint;
        else if(st.basetype==spirv_cross::SPIRType::Int)
          r.sampleType = WGPUTextureSampleType_Sint;
        else
          r.sampleType = WGPUTextureSampleType_Float;
        r.depth = t.image.depth;
        if(r.depth)
          r.sampleType = WGPUTextureSampleType_Depth;
        r.storageFmt = storageFormat(t.image.format);
        auto dec = comp.get_decoration_bitset(b.spvId);
        const bool nr = dec.get(spv::DecorationNonReadable);
        const bool nw = dec.get(spv::DecorationNonWritable);
        r.access = nr ? WGPUStorageTextureAccess_WriteOnly :
                   nw ? WGPUStorageTextureAccess_ReadOnly  : WGPUStorageTextureAccess_ReadWrite;
        }
      }
    if(stage==ShaderReflection::Fragment) {
      auto rs = comp.get_shader_resources();
      for(auto& o:rs.stage_outputs) {
        const uint32_t loc = comp.get_decoration(o.id, spv::DecorationLocation);
        if(loc>=MaxFramebufferAttachments)
          continue;
        fragOutputs |= (1u<<loc);
        auto& t = comp.get_type(o.base_type_id);
        fragUintOut[loc] = (t.basetype==spirv_cross::SPIRType::UInt || t.basetype==spirv_cross::SPIRType::Int);
        }
      }
    }
  };

// layout shared by graphics and compute pipelines
struct Layout {
  ShaderReflection::LayoutDesc lx;
  ShaderReflection::SyncDesc   sync;
  ShaderReflection::PushBlock  pb;
  ResInfo                      res[MaxBindings];

  // variant: bit i -> texture at binding i must be bound as unfilterable
  std::unordered_map<uint32_t, std::pair<WGPUBindGroupLayout, WGPUPipelineLayout>> variants;

  void init(const WShader* const* sh, size_t cnt) {
    const std::vector<ShaderReflection::Binding>* lays[3] = {};
    for(size_t i=0; i<cnt; ++i)
      lays[i] = sh[i] ? &sh[i]->lay : nullptr;
    ShaderReflection::setupLayout(pb, lx, sync, lays, cnt);
    for(size_t i=0; i<cnt; ++i) {
      if(sh[i]==nullptr)
        continue;
      for(auto& b:sh[i]->lay)
        if(b.layout<MaxBindings && b.cls!=ShaderReflection::Push)
          res[b.layout] = sh[i]->res[b.layout];
      }
    }

  static WGPUShaderStageFlags visibility(ShaderReflection::Stage s) {
    WGPUShaderStageFlags f = 0;
    if(s & ShaderReflection::Vertex)   f |= WGPUShaderStage_Vertex;
    if(s & ShaderReflection::Fragment) f |= WGPUShaderStage_Fragment;
    if(s & ShaderReflection::Compute)  f |= WGPUShaderStage_Compute;
    return f;
    }

  bool hasSeparateSampler() const {
    for(size_t i=0; i<MaxBindings; ++i)
      if((lx.active & (1u<<i)) && lx.bindings[i]==ShaderReflection::Sampler)
        return true;
    return false;
    }

  std::pair<WGPUBindGroupLayout, WGPUPipelineLayout> get(uint32_t variant) {
    auto it = variants.find(variant);
    if(it!=variants.end())
      return it->second;

    std::vector<WGPUBindGroupLayoutEntry> e;
    for(uint32_t i=0; i<MaxBindings; ++i) {
      if((lx.active & (1u<<i))==0)
        continue;
      const auto vis = visibility(lx.stage[i]);
      const bool unfilt = (variant & (1u<<i))!=0;
      WGPUBindGroupLayoutEntry x = {};
      x.binding    = i;
      x.visibility = vis;
      switch(lx.bindings[i]) {
        case ShaderReflection::Ubo:
          x.buffer.type = WGPUBufferBindingType_Uniform;
          e.push_back(x);
          break;
        case ShaderReflection::SsboR:
          x.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
          e.push_back(x);
          break;
        case ShaderReflection::SsboRW:
          x.buffer.type = WGPUBufferBindingType_Storage;
          e.push_back(x);
          break;
        case ShaderReflection::Texture: {
          x.texture.sampleType    = res[i].sampleType;
          if(unfilt && x.texture.sampleType==WGPUTextureSampleType_Float)
            x.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
          x.texture.viewDimension = res[i].dim;
          e.push_back(x);
          WGPUBindGroupLayoutEntry s = {};
          s.binding      = i + SamplerShift;
          s.visibility   = vis;
          if(res[i].depth)
            s.sampler.type = WGPUSamplerBindingType_Comparison;
          else if(unfilt || res[i].sampleType!=WGPUTextureSampleType_Float)
            s.sampler.type = WGPUSamplerBindingType_NonFiltering;
          else
            s.sampler.type = WGPUSamplerBindingType_Filtering;
          e.push_back(s);
          break;
          }
        case ShaderReflection::Image:
          x.texture.sampleType    = res[i].sampleType;
          if(unfilt && x.texture.sampleType==WGPUTextureSampleType_Float)
            x.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
          x.texture.viewDimension = res[i].dim;
          e.push_back(x);
          break;
        case ShaderReflection::Sampler:
          x.sampler.type = (variant!=0) ? WGPUSamplerBindingType_NonFiltering : WGPUSamplerBindingType_Filtering;
          e.push_back(x);
          break;
        case ShaderReflection::ImgR:
        case ShaderReflection::ImgRW:
          x.storageTexture.access        = res[i].access;
          x.storageTexture.format        = res[i].storageFmt;
          x.storageTexture.viewDimension = res[i].dim;
          e.push_back(x);
          break;
        default:
          break;
        }
      }
    if(pb.size>0) {
      WGPUBindGroupLayoutEntry x = {};
      x.binding                = PushBinding;
      x.visibility             = visibility(pb.stage);
      x.buffer.type            = WGPUBufferBindingType_Uniform;
      x.buffer.hasDynamicOffset = true;
      e.push_back(x);
      }

    WGPUBindGroupLayoutDescriptor bd = {};
    bd.entryCount = e.size();
    bd.entries    = e.data();
    auto bgl = wgpuDeviceCreateBindGroupLayout(g.dev, &bd);

    WGPUPipelineLayoutDescriptor pd = {};
    pd.bindGroupLayoutCount = 1;
    pd.bindGroupLayouts     = &bgl;
    auto pl = wgpuDeviceCreatePipelineLayout(g.dev, &pd);
    variants[variant] = {bgl, pl};
    return {bgl, pl};
    }
  };

struct WPipeline : AbstractGraphicsApi::Pipeline {
  RenderState  st;
  Topology     tp = Triangles;
  WShader*     vs = nullptr;
  WShader*     fs = nullptr;
  // the engine drops its Shader handles right after pipeline creation, while
  // pipeline instances are created lazily: keep the modules alive
  Detail::DSharedPtr<const WShader*> vsRef, fsRef;
  Layout       lay;
  bool         ok = false;
  std::vector<Decl::ComponentType> decl;
  std::unordered_map<std::string, WGPURenderPipeline> inst;

  IVec3  workGroupSize() const override { return IVec3(1,1,1); }
  size_t sizeofBuffer(size_t id, size_t arraylen) const override { return lay.lx.sizeofBuffer(id, arraylen); }

  WGPURenderPipeline instance(uint32_t variant, const WGPUTextureFormat* color, size_t colorCnt,
                              WGPUTextureFormat depth, size_t stride) {
    std::string key;
    key.reserve(32);
    auto put = [&](uint32_t v){ key.append(reinterpret_cast<const char*>(&v), 4); };
    put(variant); put(uint32_t(depth)); put(uint32_t(stride)); put(uint32_t(colorCnt));
    for(size_t i=0; i<colorCnt; ++i)
      put(uint32_t(color[i]));
    auto it = inst.find(key);
    if(it!=inst.end())
      return it->second;

    auto layouts = lay.get(variant);

    WGPUVertexAttribute attr[16] = {};
    WGPUVertexBufferLayout vbl = {};
    uint32_t offset = 0;
    for(size_t i=0; i<decl.size() && i<16; ++i) {
      attr[i].format         = vertexFormat(decl[i]);
      attr[i].offset         = offset;
      attr[i].shaderLocation = uint32_t(i);
      offset += uint32_t(Decl::size(decl[i]));
      }
    vbl.arrayStride    = stride>0 ? stride : offset;
    vbl.stepMode       = WGPUVertexStepMode_Vertex;
    vbl.attributeCount = std::min<size_t>(decl.size(), 16);
    vbl.attributes     = attr;

    WGPURenderPipelineDescriptor d = {};
    d.label  = vs->name.c_str();
    d.layout = layouts.second;
    d.vertex.module      = vs->module;
    d.vertex.entryPoint  = "main";
    d.vertex.bufferCount = decl.empty() ? 0 : 1;
    d.vertex.buffers     = &vbl;

    switch(tp) {
      case Points:    d.primitive.topology = WGPUPrimitiveTopology_PointList;    break;
      case Lines:     d.primitive.topology = WGPUPrimitiveTopology_LineList;     break;
      case Triangles: d.primitive.topology = WGPUPrimitiveTopology_TriangleList; break;
      }
    d.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    d.primitive.frontFace        = WGPUFrontFace_CW;
    switch(st.cullFaceMode()) {
      case RenderState::CullMode::Back:   d.primitive.cullMode = WGPUCullMode_Back;  break;
      case RenderState::CullMode::Front:  d.primitive.cullMode = WGPUCullMode_Front; break;
      case RenderState::CullMode::NoCull: d.primitive.cullMode = WGPUCullMode_None;  break;
      }

    WGPUDepthStencilState ds = {};
    if(depth!=WGPUTextureFormat_Undefined) {
      ds.format            = depth;
      ds.depthWriteEnabled = st.isZWriteEnabled() && !st.isRasterDiscardEnabled();
      ds.depthCompare      = compareFn(st.zTestMode());
      ds.stencilFront      = {WGPUCompareFunction_Always, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep};
      ds.stencilBack       = ds.stencilFront;
      ds.stencilReadMask   = 0xFFFFFFFF;
      ds.stencilWriteMask  = 0;
      d.depthStencil       = &ds;
      }

    d.multisample.count = 1;
    d.multisample.mask  = 0xFFFFFFFF;

    WGPUBlendState        blend = {};
    blend.color.srcFactor = blendFactor(st.blendSource());
    blend.color.dstFactor = blendFactor(st.blendDest());
    blend.color.operation = blendOp(st.blendOperation());
    blend.alpha           = blend.color;
    if(blend.color.operation==WGPUBlendOperation_Min || blend.color.operation==WGPUBlendOperation_Max) {
      blend.color.srcFactor = blend.color.dstFactor = WGPUBlendFactor_One;
      blend.alpha = blend.color;
      }

    WGPUColorTargetState ct[MaxFramebufferAttachments] = {};
    WGPUFragmentState    fst = {};
    if(fs!=nullptr && fs->module!=nullptr) {
      for(size_t i=0; i<colorCnt; ++i) {
        ct[i].format    = color[i];
        const bool has  = (fs->fragOutputs & (1u<<i))!=0;
        const bool ints = (color[i]==WGPUTextureFormat_R32Uint || color[i]==WGPUTextureFormat_RG32Uint ||
                           color[i]==WGPUTextureFormat_RGBA32Uint);
        ct[i].blend     = (st.hasBlend() && !ints) ? &blend : nullptr;
        ct[i].writeMask = (has && !st.isRasterDiscardEnabled()) ? WGPUColorWriteMask_All : WGPUColorWriteMask_None;
        }
      fst.module      = fs->module;
      fst.entryPoint  = "main";
      fst.targetCount = colorCnt;
      fst.targets     = ct;
      d.fragment      = &fst;
      }

    scopeBegin();
    auto p = wgpuDeviceCreateRenderPipeline(g.dev, &d);
    scopeEnd("pipeline " + vs->name + " / " + (fs ? fs->name : std::string("-")));
    inst[key] = p;
    return p;
    }
  };

struct WCompPipeline : AbstractGraphicsApi::CompPipeline {
  WShader* cs = nullptr;
  Detail::DSharedPtr<const WShader*> csRef;
  Layout   lay;
  IVec3    wg;
  std::unordered_map<uint32_t, WGPUComputePipeline> inst;

  IVec3  workGroupSize() const override { return wg; }
  size_t sizeofBuffer(size_t id, size_t arraylen) const override { return lay.lx.sizeofBuffer(id, arraylen); }

  WGPUComputePipeline instance(uint32_t variant) {
    auto it = inst.find(variant);
    if(it!=inst.end())
      return it->second;
    auto layouts = lay.get(variant);
    WGPUComputePipelineDescriptor d = {};
    d.label              = cs->name.c_str();
    d.layout             = layouts.second;
    d.compute.module     = cs->module;
    d.compute.entryPoint = "main";
    scopeBegin();
    auto p = wgpuDeviceCreateComputePipeline(g.dev, &d);
    scopeEnd("compute " + cs->name);
    inst[variant] = p;
    return p;
    }
  };

// ---------------------------------------------------------------- swapchain

struct WSwapchain : AbstractGraphicsApi::Swapchain {
  WGPUSurface     surface = nullptr;
  WGPUSwapChain   sc      = nullptr;
  WGPUTextureView cur     = nullptr;
  uint32_t        width = 1, height = 1;

  WSwapchain() {
    WGPUSurfaceDescriptorFromCanvasHTMLSelector cs = {};
    cs.chain.sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
    cs.selector    = "#canvas";
    WGPUSurfaceDescriptor sd = {};
    sd.nextInChain = &cs.chain;
    surface = wgpuInstanceCreateSurface(wgpuCreateInstance(nullptr), &sd);
    reset();
    }

  ~WSwapchain() override {
    if(cur!=nullptr)
      wgpuTextureViewRelease(cur);
    }

  void reset() override {
    double cssW = 0, cssH = 0;
    emscripten_get_element_css_size("#canvas", &cssW, &cssH);
    const double dpr = emscripten_get_device_pixel_ratio();
    width  = uint32_t(std::max(1.0, cssW*dpr + 0.5));
    height = uint32_t(std::max(1.0, cssH*dpr + 0.5));
    if(cur!=nullptr) {
      wgpuTextureViewRelease(cur);
      cur = nullptr;
      }
    WGPUSwapChainDescriptor d = {};
    d.usage       = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    d.format      = g.canvasFormat;
    d.width       = width;
    d.height      = height;
    d.presentMode = WGPUPresentMode_Fifo;
    sc = wgpuDeviceCreateSwapChain(g.dev, surface, &d);
    Log::i("[WebGPU] swapchain ", width, "x", height, " dpr=", dpr);
    }

  WGPUTextureView current() {
    if(cur==nullptr)
      cur = wgpuSwapChainGetCurrentTextureView(sc);
    return cur;
    }

  void presented() {
    if(cur!=nullptr) {
      wgpuTextureViewRelease(cur);
      cur = nullptr;
      }
    }

  uint32_t currentBackBufferIndex() override { return 0; }
  uint32_t imageCount() const override { return 1; }
  uint32_t w() const override { return width; }
  uint32_t h() const override { return height; }
  };

std::vector<WSwapchain*> g_swapchains;

// --------------------------------------------------------------- mip blits

const char* blitWgsl = R"(
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var smp : sampler;
struct VO { @builtin(position) pos : vec4f, @location(0) uv : vec2f };
@vertex fn vs(@builtin(vertex_index) i : u32) -> VO {
  var o : VO;
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  o.pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  o.uv  = uv;
  return o;
}
@fragment fn fs(v : VO) -> @location(0) vec4f {
  return textureSampleLevel(src, smp, v.uv, 0.0);
}
)";

WGPURenderPipeline blitPipeline(WGPUTextureFormat fmt) {
  auto it = g.blitPso.find(uint32_t(fmt));
  if(it!=g.blitPso.end())
    return it->second;
  if(g.blitModule==nullptr) {
    WGPUShaderModuleWGSLDescriptor wd = {};
    wd.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wd.code        = blitWgsl;
    WGPUShaderModuleDescriptor md = {};
    md.nextInChain = &wd.chain;
    g.blitModule = wgpuDeviceCreateShaderModule(g.dev, &md);

    WGPUBindGroupLayoutEntry e[2] = {};
    e[0].binding = 0;
    e[0].visibility = WGPUShaderStage_Fragment;
    e[0].texture.sampleType = WGPUTextureSampleType_Float;
    e[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    e[1].binding = 1;
    e[1].visibility = WGPUShaderStage_Fragment;
    e[1].sampler.type = WGPUSamplerBindingType_Filtering;
    WGPUBindGroupLayoutDescriptor bd = {};
    bd.entryCount = 2;
    bd.entries    = e;
    g.blitBgl = wgpuDeviceCreateBindGroupLayout(g.dev, &bd);
    WGPUPipelineLayoutDescriptor pd = {};
    pd.bindGroupLayoutCount = 1;
    pd.bindGroupLayouts     = &g.blitBgl;
    g.blitLay = wgpuDeviceCreatePipelineLayout(g.dev, &pd);
    }
  WGPUColorTargetState ct = {};
  ct.format    = fmt;
  ct.writeMask = WGPUColorWriteMask_All;
  WGPUFragmentState fs = {};
  fs.module      = g.blitModule;
  fs.entryPoint  = "fs";
  fs.targetCount = 1;
  fs.targets     = &ct;
  WGPURenderPipelineDescriptor d = {};
  d.layout             = g.blitLay;
  d.vertex.module      = g.blitModule;
  d.vertex.entryPoint  = "vs";
  d.primitive.topology = WGPUPrimitiveTopology_TriangleList;
  d.primitive.cullMode = WGPUCullMode_None;
  d.multisample.count  = 1;
  d.multisample.mask   = 0xFFFFFFFF;
  d.fragment           = &fs;
  auto p = wgpuDeviceCreateRenderPipeline(g.dev, &d);
  g.blitPso[uint32_t(fmt)] = p;
  return p;
  }

// ----------------------------------------------------------- bind groups

struct KeyHash {
  size_t operator()(const std::vector<uint64_t>& k) const {
    uint64_t h = 1469598103934665603ull;
    for(auto v:k) {
      h ^= v;
      h *= 1099511628211ull;
      }
    return size_t(h ^ (h>>32));
    }
  };
std::unordered_map<std::vector<uint64_t>, WGPUBindGroup, KeyHash> g_bindGroups;

void reportFirstUse(const std::string& tag) {
  static std::unordered_map<std::string, bool> seen;
  if(seen[tag])
    return;
  seen[tag] = true;
  Log::i("[WebGPU] first use: ", tag);
  }

void reportMissing(const std::string& tag, uint32_t slot) {
  static std::unordered_map<std::string, bool> seen;
  if(seen[tag])
    return;
  seen[tag] = true;
  Log::e("[WebGPU] ", tag, ": draw skipped, binding ", slot, " is not set");
  }

// -------------------------------------------------------- command buffer

struct WCommandBuffer : AbstractGraphicsApi::CommandBuffer {
  enum State : uint8_t { Idle, Render, Compute };

  WGPUCommandEncoder        enc  = nullptr;
  WGPURenderPassEncoder     rp   = nullptr;
  WGPUComputePassEncoder    cp   = nullptr;
  WGPUCommandBuffer         done = nullptr;
  bool                      recording = false;

  // current render pass
  WGPUTextureFormat         passColor[MaxFramebufferAttachments] = {};
  size_t                    passColorCnt = 0;
  WGPUTextureFormat         passDepth    = WGPUTextureFormat_Undefined;
  uint32_t                  passW = 0, passH = 0;

  WPipeline*                gp = nullptr;
  WCompPipeline*            cpso = nullptr;
  Detail::Bindings          bind;
  bool                      bindDirty = true;

  uint8_t                   push[256] = {};
  size_t                    pushSize  = 0;
  bool                      pushDirty = true;
  uint32_t                  pushOffset = 0;
  WGPUBuffer                pushBufCur = nullptr;

  struct Chunk {
    WGPUBuffer           buf = nullptr;
    std::vector<uint8_t> cpu;
    size_t               used = 0;
    };
  std::vector<Chunk>        chunks;
  size_t                    chunkId = 0;

  // last state set on the encoder
  const void*               lastPso   = nullptr;
  WGPUBindGroup             lastBg    = nullptr;
  uint32_t                  lastPushOff = uint32_t(-1);
  const WBuffer*            lastVbo   = nullptr;
  const WBuffer*            lastIbo   = nullptr;
  size_t                    vboStride = 0;

  ~WCommandBuffer() override {
    for(auto& c:chunks)
      if(c.buf!=nullptr)
        g.deadBuf.push_back(c.buf);
    }

  bool isRecording() const override { return recording; }

  void begin() override {
    reset();
    WGPUCommandEncoderDescriptor d = {};
    enc       = wgpuDeviceCreateCommandEncoder(g.dev, &d);
    recording = true;
    chunkId   = 0;
    for(auto& c:chunks)
      c.used = 0;
    }

  void end() override {
    closePass();
    if(enc!=nullptr) {
      WGPUCommandBufferDescriptor d = {};
      done = wgpuCommandEncoderFinish(enc, &d);
      wgpuCommandEncoderRelease(enc);
      enc = nullptr;
      }
    recording = false;
    }

  void reset() override {
    closePass();
    if(enc!=nullptr) {
      wgpuCommandEncoderRelease(enc);
      enc = nullptr;
      }
    if(done!=nullptr) {
      wgpuCommandBufferRelease(done);
      done = nullptr;
      }
    recording = false;
    gp   = nullptr;
    cpso = nullptr;
    bind = Detail::Bindings();
    bindDirty = true;
    pushDirty = true;
    resetEncoderState();
    }

  void resetEncoderState() {
    lastPso     = nullptr;
    lastBg      = nullptr;
    lastPushOff = uint32_t(-1);
    lastVbo     = nullptr;
    lastIbo     = nullptr;
    }

  void closePass() {
    if(rp!=nullptr) {
      wgpuRenderPassEncoderEnd(rp);
      wgpuRenderPassEncoderRelease(rp);
      rp = nullptr;
      }
    if(cp!=nullptr) {
      wgpuComputePassEncoderEnd(cp);
      wgpuComputePassEncoderRelease(cp);
      cp = nullptr;
      }
    resetEncoderState();
    }

  void submitTo(WGPUQueue q) {
    for(size_t i=0; i<chunks.size() && i<=chunkId; ++i) {
      auto& c = chunks[i];
      if(c.used>0)
        wgpuQueueWriteBuffer(q, c.buf, 0, c.cpu.data(), alignUp(uint64_t(c.used), uint64_t(4)));
      }
    if(done!=nullptr) {
      ++g.st.submits;
      wgpuQueueSubmit(q, 1, &done);
      wgpuCommandBufferRelease(done);
      done = nullptr;
      }
    }

  // ------------------------------------------------------------ passes
  void beginRendering(const AttachmentDesc* desc, size_t descSize, uint32_t w, uint32_t h,
                      const TextureFormat* frm, AbstractGraphicsApi::Texture** att,
                      AbstractGraphicsApi::Swapchain** sw, const uint32_t* imgId) override {
    (void)imgId;
    closePass();

    WGPURenderPassColorAttachment        ca[MaxFramebufferAttachments] = {};
    WGPURenderPassDepthStencilAttachment da = {};
    bool hasDepth = false;
    passColorCnt = 0;
    passDepth    = WGPUTextureFormat_Undefined;

    for(size_t i=0; i<descSize; ++i) {
      const bool isDepth = (sw[i]==nullptr) && isDepthFormat(frm[i]);
      if(isDepth) {
        auto& t = *static_cast<WTexture*>(att[i]);
        hasDepth = true;
        passDepth = t.info.wgpu;
        da.view = t.attachmentView(0);
        if(desc[i].load==AccessOp::Readonly) {
          da.depthReadOnly = true;
          da.depthLoadOp   = WGPULoadOp_Undefined;
          da.depthStoreOp  = WGPUStoreOp_Undefined;
          } else {
          da.depthLoadOp     = desc[i].load==AccessOp::Preserve ? WGPULoadOp_Load : WGPULoadOp_Clear;
          da.depthStoreOp    = desc[i].store==AccessOp::Preserve ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
          da.depthClearValue = desc[i].load==AccessOp::Clear ? desc[i].clear.x : 1.f;
          }
        if(t.frm==TextureFormat::Depth24S8) {
          da.stencilLoadOp  = da.depthReadOnly ? WGPULoadOp_Undefined  : WGPULoadOp_Clear;
          da.stencilStoreOp = da.depthReadOnly ? WGPUStoreOp_Undefined : WGPUStoreOp_Discard;
          da.stencilReadOnly = da.depthReadOnly;
          } else {
          da.stencilLoadOp  = WGPULoadOp_Undefined;
          da.stencilStoreOp = WGPUStoreOp_Undefined;
          }
        continue;
        }
      auto& c = ca[passColorCnt];
      const bool probe = (sw[i]!=nullptr) && (g.dbg & 2);
      if(sw[i]!=nullptr) {
        auto& s = *static_cast<WSwapchain*>(sw[i]);
        c.view = s.current();
        passColor[passColorCnt] = g.canvasFormat;
        } else {
        auto& t = *static_cast<WTexture*>(att[i]);
        c.view = t.attachmentView(0);
        passColor[passColorCnt] = t.info.wgpu;
        }
      c.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
      c.loadOp     = desc[i].load==AccessOp::Preserve ? WGPULoadOp_Load : WGPULoadOp_Clear;
      c.storeOp    = desc[i].store==AccessOp::Discard ? WGPUStoreOp_Discard : WGPUStoreOp_Store;
      if(desc[i].load==AccessOp::Clear)
        c.clearValue = {desc[i].clear.x, desc[i].clear.y, desc[i].clear.z, desc[i].clear.w};
      else
        c.clearValue = {0, 0, 0, 0};
      if(probe) {
        c.loadOp     = WGPULoadOp_Clear;
        c.clearValue = {1, 0, 1, 1};
        }
      ++passColorCnt;
      }

    WGPURenderPassDescriptor d = {};
    d.colorAttachmentCount   = passColorCnt;
    d.colorAttachments       = ca;
    d.depthStencilAttachment = hasDepth ? &da : nullptr;
    rp    = wgpuCommandEncoderBeginRenderPass(enc, &d);
    ++g.st.passes;
    for(size_t i=0; i<descSize; ++i)
      if(sw[i]!=nullptr)
        ++g.st.screenPasses;
    passW = w;
    passH = h;
    resetEncoderState();
    }

  void endRendering() override {
    closePass();
    }

  void barrier(const AbstractGraphicsApi::SyncDesc&, const AbstractGraphicsApi::BarrierDesc*, size_t) override {}

  void ensureCompute() {
    if(cp!=nullptr)
      return;
    closePass();
    WGPUComputePassDescriptor d = {};
    cp = wgpuCommandEncoderBeginComputePass(enc, &d);
    resetEncoderState();
    }

  // ------------------------------------------------------------ state
  void setPipeline(AbstractGraphicsApi::Pipeline& p) override {
    gp = static_cast<WPipeline*>(&p);
    bindDirty = true;
    pushDirty = true;
    }

  void setComputePipeline(AbstractGraphicsApi::CompPipeline& p) override {
    cpso = static_cast<WCompPipeline*>(&p);
    bindDirty = true;
    pushDirty = true;
    }

  void setPushData(const void* data, size_t size) override {
    size = std::min(size, sizeof(push));
    std::memcpy(push, data, size);
    pushSize  = size;
    pushDirty = true;
    }

  void setBinding(size_t id, AbstractGraphicsApi::Texture* tex, uint32_t mipLevel,
                  const ComponentMapping& m, const Sampler& smp) override {
    bind.data  [id] = tex;
    bind.smp   [id] = smp;
    bind.map   [id] = m;
    bind.offset[id] = mipLevel;
    bind.array &= ~(1u<<id);
    bindDirty = true;
    }

  void setBinding(size_t id, AbstractGraphicsApi::Buffer* buf, size_t offset) override {
    bind.data  [id] = buf;
    bind.offset[id] = uint32_t(offset);
    bind.array &= ~(1u<<id);
    bindDirty = true;
    }

  void setBinding(size_t id, AbstractGraphicsApi::DescArray* arr) override {
    bind.data[id] = arr;
    bind.array |= (1u<<id);
    bindDirty = true;
    }

  void setBinding(size_t, AbstractGraphicsApi::AccelerationStructure*) override {}

  void setBinding(size_t id, const Sampler& smp) override {
    bind.smp[id] = smp;
    bindDirty = true;
    }

  void setViewport(const Rect& r) override {
    if(rp==nullptr)
      return;
    float x = float(r.x), y = float(r.y), w = float(r.w), h = float(r.h);
    if(w<=0 || h<=0)
      return;
    wgpuRenderPassEncoderSetViewport(rp, x, y, w, h, 0.f, 1.f);
    }

  void setScissor(const Rect& r) override {
    if(rp==nullptr)
      return;
    int x0 = std::max(0, r.x), y0 = std::max(0, r.y);
    int x1 = std::min(int(passW), r.x+r.w), y1 = std::min(int(passH), r.y+r.h);
    if(x1<x0) x1 = x0;
    if(y1<y0) y1 = y0;
    wgpuRenderPassEncoderSetScissorRect(rp, uint32_t(x0), uint32_t(y0), uint32_t(x1-x0), uint32_t(y1-y0));
    }

  // ------------------------------------------------------------ bindings
  uint32_t variantOf(const Layout& lay) const {
    uint32_t v = 0;
    for(uint32_t i=0; i<MaxBindings; ++i) {
      if((lay.lx.active & (1u<<i))==0)
        continue;
      auto cls = lay.lx.bindings[i];
      if(cls!=ShaderReflection::Texture && cls!=ShaderReflection::Image)
        continue;
      if(lay.res[i].sampleType!=WGPUTextureSampleType_Float)
        continue;
      auto* t = static_cast<WTexture*>(static_cast<AbstractGraphicsApi::Texture*>(bind.data[i]));
      if(t==nullptr || (bind.array & (1u<<i)))
        continue;
      if(t->info.depth || (t->info.float32 && !g.f32filter) || t->info.integer)
        v |= (1u<<i);
      }
    return v;
    }

  uint32_t pushOffsetFor(const Layout& lay) {
    if(lay.pb.size==0)
      return 0;
    if(!pushDirty)
      return pushOffset;
    const size_t sz = alignUp(uint32_t(std::max<size_t>(lay.pb.size, 16)), PushAlign);
    if(chunks.empty() || chunks[chunkId].used + sz > PushChunk) {
      if(!chunks.empty() && chunks[chunkId].used>0)
        ++chunkId;
      if(chunkId>=chunks.size()) {
        Chunk c;
        WGPUBufferDescriptor d = {};
        d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        d.size  = PushChunk;
        c.buf   = wgpuDeviceCreateBuffer(g.dev, &d);
        c.cpu.resize(PushChunk);
        chunks.push_back(std::move(c));
        }
      }
    auto& c = chunks[chunkId];
    std::memset(c.cpu.data()+c.used, 0, sz);
    std::memcpy(c.cpu.data()+c.used, push, std::min<size_t>(pushSize, lay.pb.size));
    pushOffset = uint32_t(c.used);
    pushBufCur = c.buf;
    c.used += sz;
    pushDirty = false;
    return pushOffset;
    }

  WGPUBindGroup bindGroup(Layout& lay, uint32_t variant, WGPUBindGroupLayout bgl, const std::string& tag) {
    std::vector<uint64_t>           key;
    std::vector<WGPUBindGroupEntry> e;
    key.reserve(64);
    key.push_back(reinterpret_cast<uintptr_t>(bgl));
    const bool sepNonFilt = variant!=0;

    for(uint32_t i=0; i<MaxBindings; ++i) {
      if((lay.lx.active & (1u<<i))==0)
        continue;
      auto cls  = lay.lx.bindings[i];
      auto* res = bind.data[i];
      WGPUBindGroupEntry x = {};
      x.binding = i;
      switch(cls) {
        case ShaderReflection::Ubo:
        case ShaderReflection::SsboR:
        case ShaderReflection::SsboRW: {
          auto* b = static_cast<WBuffer*>(static_cast<AbstractGraphicsApi::Buffer*>(res));
          if(b==nullptr || (bind.array & (1u<<i))) {
            reportMissing(tag, i);
            return nullptr;
            }
          uint64_t off = bind.offset[i];
          uint64_t sz  = b->alloc>off ? b->alloc-off : 0;
          if(cls==ShaderReflection::Ubo)
            sz = std::min<uint64_t>(sz, g.limits.limits.maxUniformBufferBindingSize);
          if(sz==0)
            return nullptr;
          x.buffer = b->impl;
          x.offset = off;
          x.size   = sz;
          key.push_back(b->uid); key.push_back(off);
          e.push_back(x);
          break;
          }
        case ShaderReflection::Texture:
        case ShaderReflection::Image: {
          auto* t = static_cast<WTexture*>(static_cast<AbstractGraphicsApi::Texture*>(res));
          if(t==nullptr || (bind.array & (1u<<i))) {
            reportMissing(tag, i);
            return nullptr;
            }
          uint32_t mip = bind.offset[i];
          x.textureView = t->view(mip, lay.res[i].dim);
          key.push_back(t->uid); key.push_back(uint64_t(mip) | uint64_t(lay.res[i].dim)<<32);
          e.push_back(x);
          if(cls==ShaderReflection::Texture) {
            const bool unfilt = (variant & (1u<<i))!=0 || lay.res[i].sampleType!=WGPUTextureSampleType_Float;
            WGPUBindGroupEntry s = {};
            s.binding = i + SamplerShift;
            s.sampler = sampler(bind.smp[i], unfilt && !lay.res[i].depth, lay.res[i].depth);
            key.push_back(reinterpret_cast<uintptr_t>(s.sampler));
            e.push_back(s);
            }
          break;
          }
        case ShaderReflection::Sampler: {
          x.sampler = sampler(bind.smp[i], sepNonFilt, false);
          key.push_back(reinterpret_cast<uintptr_t>(x.sampler));
          e.push_back(x);
          break;
          }
        case ShaderReflection::ImgR:
        case ShaderReflection::ImgRW: {
          auto* t = static_cast<WTexture*>(static_cast<AbstractGraphicsApi::Texture*>(res));
          if(t==nullptr) {
            reportMissing(tag, i);
            return nullptr;
            }
          uint32_t mip = bind.offset[i]==uint32_t(-1) ? 0 : bind.offset[i];
          x.textureView = t->view(mip, lay.res[i].dim);
          key.push_back(t->uid); key.push_back(uint64_t(mip) | uint64_t(lay.res[i].dim)<<32);
          e.push_back(x);
          break;
          }
        default:
          break;
        }
      }
    if(lay.pb.size>0) {
      WGPUBindGroupEntry x = {};
      x.binding = PushBinding;
      x.buffer  = pushBufCur;
      x.offset  = 0;
      x.size    = alignUp(uint64_t(std::max<uint32_t>(lay.pb.size, 16)), uint64_t(16));
      key.push_back(reinterpret_cast<uintptr_t>(pushBufCur));
      e.push_back(x);
      }

    auto it = g_bindGroups.find(key);
    if(it!=g_bindGroups.end())
      return it->second;
    if(g_bindGroups.size()>16384) {
      for(auto& i:g_bindGroups)
        wgpuBindGroupRelease(i.second);
      g_bindGroups.clear();
      }
    WGPUBindGroupDescriptor d = {};
    d.layout     = bgl;
    d.entryCount = e.size();
    d.entries    = e.data();
    scopeBegin();
    auto bg = wgpuDeviceCreateBindGroup(g.dev, &d);
    scopeEnd("bindgroup " + tag);
    g_bindGroups[std::move(key)] = bg;
    return bg;
    }

  bool flushGraphics(size_t stride) {
    ++g.st.draws;
    if(gp==nullptr || !gp->ok || rp==nullptr) {
      ++g.st.skipped;
      return false;
      }
    reportFirstUse("draw " + gp->vs->name + " / " + (gp->fs ? gp->fs->name : std::string("-")));
    auto& lay     = gp->lay;
    const uint32_t variant = variantOf(lay);
    auto layouts  = lay.get(variant);
    auto pso      = gp->instance(variant, passColor, passColorCnt, passDepth, stride);
    if(pso==nullptr)
      return false;
    if(lastPso!=pso) {
      wgpuRenderPassEncoderSetPipeline(rp, pso);
      lastPso = pso;
      }
    const uint32_t off = pushOffsetFor(lay);
    auto bg = bindGroup(lay, variant, layouts.first, gp->vs->name);
    if(bg==nullptr)
      return false;
    if(bg!=lastBg || (lay.pb.size>0 && off!=lastPushOff)) {
      wgpuRenderPassEncoderSetBindGroup(rp, 0, bg, lay.pb.size>0 ? 1 : 0, &off);
      lastBg      = bg;
      lastPushOff = off;
      }
    return true;
    }

  bool flushCompute() {
    ++g.st.dispatches;
    if(cpso==nullptr || cpso->cs==nullptr || cpso->cs->module==nullptr)
      return false;
    ensureCompute();
    reportFirstUse("dispatch " + cpso->cs->name);
    auto& lay    = cpso->lay;
    const uint32_t variant = variantOf(lay);
    auto layouts = lay.get(variant);
    auto pso     = cpso->instance(variant);
    if(lastPso!=pso) {
      wgpuComputePassEncoderSetPipeline(cp, pso);
      lastPso = pso;
      }
    const uint32_t off = pushOffsetFor(lay);
    auto bg = bindGroup(lay, variant, layouts.first, cpso->cs->name);
    if(bg==nullptr)
      return false;
    if(bg!=lastBg || (lay.pb.size>0 && off!=lastPushOff)) {
      wgpuComputePassEncoderSetBindGroup(cp, 0, bg, lay.pb.size>0 ? 1 : 0, &off);
      lastBg      = bg;
      lastPushOff = off;
      }
    return true;
    }

  void bindVbo(const AbstractGraphicsApi::Buffer* vbo) {
    auto* b = static_cast<const WBuffer*>(vbo);
    if(b==nullptr || b==lastVbo)
      return;
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, b->impl, 0, b->alloc);
    lastVbo = b;
    }

  // ------------------------------------------------------------ draws
  void draw(const AbstractGraphicsApi::Buffer* vbo, size_t stride, size_t offset,
            size_t vertexCount, size_t firstInstance, size_t instanceCount) override {
    if(vertexCount==0 || instanceCount==0)
      return;
    if(!flushGraphics(vbo ? stride : 0))
      return;
    if(vbo!=nullptr && !gp->decl.empty())
      bindVbo(vbo);
    wgpuRenderPassEncoderDraw(rp, uint32_t(vertexCount), uint32_t(instanceCount),
                              uint32_t(offset), uint32_t(firstInstance));
    }

  void drawIndexed(const AbstractGraphicsApi::Buffer* vbo, size_t stride, size_t voffset,
                   const AbstractGraphicsApi::Buffer& ibo, Detail::IndexClass cls, size_t ioffset,
                   size_t isize, size_t firstInstance, size_t instanceCount) override {
    if(isize==0 || instanceCount==0)
      return;
    if(!flushGraphics(vbo ? stride : 0))
      return;
    if(vbo!=nullptr && !gp->decl.empty())
      bindVbo(vbo);
    auto& ib = static_cast<const WBuffer&>(ibo);
    if(lastIbo!=&ib) {
      wgpuRenderPassEncoderSetIndexBuffer(rp, ib.impl,
                                          cls==Detail::IndexClass::i32 ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16,
                                          0, ib.alloc);
      lastIbo = &ib;
      }
    wgpuRenderPassEncoderDrawIndexed(rp, uint32_t(isize), uint32_t(instanceCount), uint32_t(ioffset),
                                     int32_t(voffset), uint32_t(firstInstance));
    }

  void drawIndirect(const AbstractGraphicsApi::Buffer& indirect, size_t offset) override {
    if(!flushGraphics(0))
      return;
    auto& b = static_cast<const WBuffer&>(indirect);
    wgpuRenderPassEncoderDrawIndirect(rp, b.impl, offset);
    }

  void dispatch(size_t x, size_t y, size_t z) override {
    if(x==0 || y==0 || z==0)
      return;
    if(!flushCompute())
      return;
    wgpuComputePassEncoderDispatchWorkgroups(cp, uint32_t(x), uint32_t(y), uint32_t(z));
    }

  void dispatchIndirect(const AbstractGraphicsApi::Buffer& indirect, size_t offset) override {
    if(!flushCompute())
      return;
    auto& b = static_cast<const WBuffer&>(indirect);
    wgpuComputePassEncoderDispatchWorkgroupsIndirect(cp, b.impl, offset);
    }

  // ------------------------------------------------------------ transfers
  void generateMipmap(AbstractGraphicsApi::Texture& tex, uint32_t, uint32_t, uint32_t mipLevels) override {
    auto& t = static_cast<WTexture&>(tex);
    genMips(enc, t, mipLevels);
    resetEncoderState();
    }

  static void genMips(WGPUCommandEncoder enc, WTexture& t, uint32_t mipLevels) {
    if(!t.info.renderable || t.info.integer || t.info.depth || t.is3D)
      return;
    if(t.info.float32 && !g.f32filter)
      return;
    auto pso = blitPipeline(t.info.wgpu);
    Sampler s = Sampler::bilinear(ClampMode::ClampToEdge);
    auto smp = sampler(s, false, false);
    for(uint32_t i=1; i<std::min(mipLevels, t.mips); ++i) {
      WGPUBindGroupEntry e[2] = {};
      e[0].binding     = 0;
      e[0].textureView = t.view(i-1, WGPUTextureViewDimension_2D);
      e[1].binding     = 1;
      e[1].sampler     = smp;
      WGPUBindGroupDescriptor bd = {};
      bd.layout     = g.blitBgl;
      bd.entryCount = 2;
      bd.entries    = e;
      auto bg = wgpuDeviceCreateBindGroup(g.dev, &bd);

      WGPURenderPassColorAttachment ca = {};
      ca.view       = t.attachmentView(i);
      ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
      ca.loadOp     = WGPULoadOp_Clear;
      ca.storeOp    = WGPUStoreOp_Store;
      WGPURenderPassDescriptor rd = {};
      rd.colorAttachmentCount = 1;
      rd.colorAttachments     = &ca;
      auto pass = wgpuCommandEncoderBeginRenderPass(enc, &rd);
      wgpuRenderPassEncoderSetPipeline(pass, pso);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass);
      wgpuRenderPassEncoderRelease(pass);
      wgpuBindGroupRelease(bg);
      }
    }

  void copy(AbstractGraphicsApi::Buffer& dst, size_t offset, AbstractGraphicsApi::Texture& src,
            uint32_t width, uint32_t height, uint32_t mip) override {
    closePass();
    auto& b = static_cast<WBuffer&>(dst);
    auto& t = static_cast<WTexture&>(src);
    const uint32_t w = std::max(1u, width>>mip), h = std::max(1u, height>>mip);
    const uint32_t row    = w*t.info.bpp;
    const uint32_t rowPad = alignUp(row, 256u);

    WGPUImageCopyTexture s = {};
    s.texture  = t.impl;
    s.mipLevel = mip;
    s.aspect   = t.info.depth ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
    WGPUExtent3D ext = {w, h, 1};

    if(row==rowPad && offset%4==0) {
      WGPUImageCopyBuffer d = {};
      d.buffer             = b.impl;
      d.layout.offset      = offset;
      d.layout.bytesPerRow = row;
      d.layout.rowsPerImage = h;
      wgpuCommandEncoderCopyTextureToBuffer(enc, &s, &d, &ext);
      return;
      }
    WGPUBufferDescriptor bd = {};
    bd.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    bd.size  = uint64_t(rowPad)*h;
    auto tmp = wgpuDeviceCreateBuffer(g.dev, &bd);
    WGPUImageCopyBuffer d = {};
    d.buffer              = tmp;
    d.layout.offset       = 0;
    d.layout.bytesPerRow  = rowPad;
    d.layout.rowsPerImage = h;
    wgpuCommandEncoderCopyTextureToBuffer(enc, &s, &d, &ext);
    for(uint32_t y=0; y<h; ++y)
      wgpuCommandEncoderCopyBufferToBuffer(enc, tmp, uint64_t(y)*rowPad, b.impl, offset + uint64_t(y)*row, row);
    g.deadBuf.push_back(tmp);
    }
  };

struct WFence : AbstractGraphicsApi::Fence {
  void wait() override {}
  bool wait(uint64_t) override { return true; }
  };

} // namespace

// ============================================================================

struct WebGapi::Impl {};

WebGapi::WebGapi(ApiFlags) {
  g.dev = emscripten_webgpu_get_device();
  if(g.dev==nullptr) {
    Log::e("[WebGPU] no preinitialized device (navigator.gpu unavailable?)");
    throw std::system_error(Tempest::GraphicsErrc::NoDevice);
    }
  g.queue = wgpuDeviceGetQueue(g.dev);
  wgpuDeviceSetUncapturedErrorCallback(g.dev, onDeviceError, nullptr);
  wgpuDeviceGetLimits(g.dev, &g.limits);
  g.bc            = wgpuDeviceHasFeature(g.dev, WGPUFeatureName_TextureCompressionBC);
  g.f32filter     = wgpuDeviceHasFeature(g.dev, WGPUFeatureName_Float32Filterable);
  g.rg11b10render = wgpuDeviceHasFeature(g.dev, WGPUFeatureName_RG11B10UfloatRenderable);
  g.firstInstance = wgpuDeviceHasFeature(g.dev, WGPUFeatureName_IndirectFirstInstance);
  g.canvasFormat  = WGPUTextureFormat(EM_ASM_INT({
    const f = navigator.gpu.getPreferredCanvasFormat();
    return f==='rgba8unorm' ? 0x12 : 0x17;
    }));
  Log::i("[WebGPU] device ready: bc=", g.bc, " f32filter=", g.f32filter,
         " rg11b10render=", g.rg11b10render, " firstInstance=", g.firstInstance,
         " maxStorageBuffers=", g.limits.limits.maxStorageBuffersPerShaderStage,
         " maxStorageTextures=", g.limits.limits.maxStorageTexturesPerShaderStage);
  }

WebGapi::~WebGapi() = default;

std::vector<AbstractGraphicsApi::Props> WebGapi::devices() const {
  std::vector<Props> dev(1);
  dev[0].type = DeviceType::Discrete;
  std::snprintf(dev[0].name, sizeof(dev[0].name), "WebGPU");
  return dev;
  }

AbstractGraphicsApi::Device* WebGapi::createDevice(std::string_view) {
  return new WDevice();
  }

AbstractGraphicsApi::Swapchain* WebGapi::createSwapchain(SystemApi::Window*, AbstractGraphicsApi::Device*) {
  auto s = new WSwapchain();
  g_swapchains.push_back(s);
  return s;
  }

AbstractGraphicsApi::PPipeline WebGapi::createPipeline(Device*, const RenderState& st, Topology tp,
                                                      const Shader* const* sh, size_t cnt) {
  std::unique_ptr<WPipeline> p(new WPipeline());
  p->st = st;
  p->tp = tp;
  const WShader* stages[3] = {};
  for(size_t i=0; i<cnt; ++i) {
    auto* s = static_cast<const WShader*>(sh[i]);
    if(s==nullptr)
      continue;
    if(s->stage==ShaderReflection::Vertex) {
      p->vs = const_cast<WShader*>(s);
      stages[0] = s;
      }
    else if(s->stage==ShaderReflection::Fragment) {
      p->fs = const_cast<WShader*>(s);
      stages[1] = s;
      }
    }
  if(p->vs==nullptr || p->vs->module==nullptr || (p->fs!=nullptr && p->fs->module==nullptr)) {
    Log::e("[WebGPU] unsupported graphics pipeline: ",
           p->vs ? p->vs->name : std::string("<no vs>"), " / ",
           p->fs ? p->fs->name : std::string("<no fs>"));
    return PPipeline(p.release());
    }
  p->vsRef = Detail::DSharedPtr<const WShader*>(p->vs);
  if(p->fs!=nullptr)
    p->fsRef = Detail::DSharedPtr<const WShader*>(p->fs);
  p->decl = p->vs->vert.decl;
  p->lay.init(stages, 2);
  p->ok = true;
  return PPipeline(p.release());
  }

AbstractGraphicsApi::PCompPipeline WebGapi::createComputePipeline(Device*, Shader* sh) {
  std::unique_ptr<WCompPipeline> p(new WCompPipeline());
  auto* s = static_cast<WShader*>(sh);
  p->cs    = s;
  p->csRef = Detail::DSharedPtr<const WShader*>(s);
  p->wg    = s->comp.wgSize;
  const WShader* stages[1] = {s};
  p->lay.init(stages, 1);
  if(s->module==nullptr)
    Log::e("[WebGPU] unsupported compute pipeline: ", s->name);
  return PCompPipeline(p.release());
  }

AbstractGraphicsApi::PShader WebGapi::createShader(Device*, const void* source, size_t size) {
  auto* src = static_cast<const uint8_t*>(source);
  if(size>=8 && std::memcmp(src, "WGSL", 4)==0) {
    uint32_t spvSize = 0;
    std::memcpy(&spvSize, src+4, 4);
    if(8+size_t(spvSize)>size || spvSize%4!=0)
      throw std::system_error(Tempest::GraphicsErrc::InvalidShaderModule);
    std::vector<uint32_t> spv(spvSize/4);
    std::memcpy(spv.data(), src+8, spvSize);
    size_t   at      = 8+spvSize;
    uint32_t nameLen = 0;
    std::memcpy(&nameLen, src+at, 4);
    at += 4;
    std::string name(reinterpret_cast<const char*>(src+at), nameLen);
    at += nameLen;
    if(at>size)
      throw std::system_error(Tempest::GraphicsErrc::InvalidShaderModule);
    const char* wgsl = reinterpret_cast<const char*>(src+at);
    return PShader(new WShader(spv.data(), spv.size(), wgsl, size-at, name));
    }
  // untranslatable module: reflection only, pipelines using it are skipped
  if(size%4!=0)
    throw std::system_error(Tempest::GraphicsErrc::InvalidShaderModule);
  std::vector<uint32_t> spv(size/4);
  std::memcpy(spv.data(), src, size);
  return PShader(new WShader(spv.data(), spv.size(), nullptr, 0, std::string()));
  }

AbstractGraphicsApi::DescArray* WebGapi::createDescriptors(Device*, AbstractGraphicsApi::Texture**, size_t, uint32_t, const Sampler&) {
  return new WDescArray();
  }

AbstractGraphicsApi::DescArray* WebGapi::createDescriptors(Device*, AbstractGraphicsApi::Texture**, size_t, uint32_t) {
  return new WDescArray();
  }

AbstractGraphicsApi::DescArray* WebGapi::createDescriptors(Device*, AbstractGraphicsApi::Buffer**, size_t) {
  return new WDescArray();
  }

AbstractGraphicsApi::PBuffer WebGapi::createBuffer(Device*, const void* mem, size_t size, MemUsage, BufferHeap heap) {
  return PBuffer(new WBuffer(mem, size, heap));
  }

static WGPUTextureUsageFlags textureUsage(const FormatInfo& info, bool storage) {
  WGPUTextureUsageFlags u = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc;
  if(info.renderable)
    u |= WGPUTextureUsage_RenderAttachment;
  if(storage && info.storage)
    u |= WGPUTextureUsage_StorageBinding;
  return u;
  }

AbstractGraphicsApi::PTexture WebGapi::createTexture(Device*, const Pixmap& p, TextureFormat frm, uint32_t mips) {
  std::unique_ptr<WTexture> t(new WTexture());
  const TextureFormat pfrm = p.format();
  t->frm  = frm;
  t->w    = std::max(1u, p.w());
  t->h    = std::max(1u, p.h());
  t->mips = std::max(1u, mips);

  // single/dual channel images are stored swizzled in RGBA8 (see VectorImage:
  // R8 -> rrr1, RG8 -> rrrg); WebGPU has no component swizzle in core
  const bool expand = (pfrm==TextureFormat::R8 || pfrm==TextureFormat::RG8);
  t->info = formatInfo(expand ? TextureFormat::RGBA8 : pfrm);
  if(pfrm==TextureFormat::RGB8)
    t->info = formatInfo(TextureFormat::RGBA8);
  if(t->info.compressed && t->mips>p.mipCount())
    t->mips = p.mipCount();
  t->create(textureUsage(t->info, false) & (t->info.compressed ? ~WGPUTextureUsageFlags(WGPUTextureUsage_RenderAttachment) : ~0u));

  const uint8_t* src = static_cast<const uint8_t*>(p.data());
  std::vector<uint8_t> cvt;
  const size_t n = size_t(p.w())*p.h();
  if(pfrm==TextureFormat::R8) {
    cvt.resize(n*4);
    for(size_t i=0; i<n; ++i) {
      cvt[i*4+0] = src[i]; cvt[i*4+1] = src[i]; cvt[i*4+2] = src[i]; cvt[i*4+3] = 255;
      }
    src = cvt.data();
    }
  else if(pfrm==TextureFormat::RG8) {
    cvt.resize(n*4);
    for(size_t i=0; i<n; ++i) {
      cvt[i*4+0] = src[i*2]; cvt[i*4+1] = src[i*2]; cvt[i*4+2] = src[i*2]; cvt[i*4+3] = src[i*2+1];
      }
    src = cvt.data();
    }
  else if(pfrm==TextureFormat::RGB8) {
    cvt.resize(n*4);
    for(size_t i=0; i<n; ++i) {
      cvt[i*4+0] = src[i*3]; cvt[i*4+1] = src[i*3+1]; cvt[i*4+2] = src[i*3+2]; cvt[i*4+3] = 255;
      }
    src = cvt.data();
    }
  else if(pfrm==TextureFormat::R16 || pfrm==TextureFormat::RG16) {
    const size_t ch = (pfrm==TextureFormat::R16 ? 1 : 2);
    cvt.resize(n*ch*4);
    auto* s16 = reinterpret_cast<const uint16_t*>(src);
    auto* f32 = reinterpret_cast<float*>(cvt.data());
    for(size_t i=0; i<n*ch; ++i)
      f32[i] = float(s16[i])/65535.f;
    src = cvt.data();
    }

  if(t->info.compressed) {
    uint32_t w = t->w, h = t->h;
    size_t   off = 0;
    for(uint32_t m=0; m<t->mips; ++m) {
      const uint32_t bw = std::max(1u, (w+3)/4), bh = std::max(1u, (h+3)/4);
      const size_t   sz = size_t(bw)*bh*t->info.bpp;
      WGPUImageCopyTexture dst = {};
      dst.texture  = t->impl;
      dst.mipLevel = m;
      dst.aspect   = WGPUTextureAspect_All;
      WGPUTextureDataLayout dl = {};
      dl.offset       = 0;
      dl.bytesPerRow  = bw*t->info.bpp;
      dl.rowsPerImage = bh;
      WGPUExtent3D ext = {bw*4, bh*4, 1};
      wgpuQueueWriteTexture(g.queue, &dst, src+off, sz, &dl, &ext);
      off += sz;
      w = std::max(1u, w/2);
      h = std::max(1u, h/2);
      }
    return PTexture(t.release());
    }

  WGPUImageCopyTexture dst = {};
  dst.texture  = t->impl;
  dst.mipLevel = 0;
  dst.aspect   = WGPUTextureAspect_All;
  WGPUTextureDataLayout dl = {};
  dl.bytesPerRow  = t->w*t->info.bpp;
  dl.rowsPerImage = t->h;
  WGPUExtent3D ext = {t->w, t->h, 1};
  wgpuQueueWriteTexture(g.queue, &dst, src, size_t(dl.bytesPerRow)*t->h, &dl, &ext);

  if(t->mips>1) {
    WGPUCommandEncoderDescriptor ed = {};
    auto enc = wgpuDeviceCreateCommandEncoder(g.dev, &ed);
    WCommandBuffer::genMips(enc, *t, t->mips);
    WGPUCommandBufferDescriptor cd = {};
    auto cmd = wgpuCommandEncoderFinish(enc, &cd);
    wgpuQueueSubmit(g.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    }
  return PTexture(t.release());
  }

AbstractGraphicsApi::PTexture WebGapi::createTexture(Device*, uint32_t w, uint32_t h, uint32_t mips, TextureFormat frm) {
  std::unique_ptr<WTexture> t(new WTexture());
  t->frm  = frm;
  t->info = formatInfo(frm);
  t->w    = std::max(1u, w);
  t->h    = std::max(1u, h);
  t->mips = std::max(1u, mips);
  t->create(textureUsage(t->info, true));
  return PTexture(t.release());
  }

AbstractGraphicsApi::PTexture WebGapi::createStorage(Device*, uint32_t w, uint32_t h, uint32_t mips, TextureFormat frm) {
  std::unique_ptr<WTexture> t(new WTexture());
  t->frm  = frm;
  t->info = formatInfo(frm);
  t->w    = std::max(1u, w);
  t->h    = std::max(1u, h);
  t->mips = std::max(1u, mips);
  t->create(textureUsage(t->info, true));
  return PTexture(t.release());
  }

AbstractGraphicsApi::PTexture WebGapi::createStorage(Device*, uint32_t w, uint32_t h, uint32_t d, uint32_t mips, TextureFormat frm) {
  std::unique_ptr<WTexture> t(new WTexture());
  t->frm  = frm;
  t->info = formatInfo(frm);
  t->w    = std::max(1u, w);
  t->h    = std::max(1u, h);
  t->d    = std::max(1u, d);
  t->is3D = true;
  t->mips = std::max(1u, mips);
  auto info = t->info;
  info.renderable = false;
  t->create(textureUsage(info, true));
  return PTexture(t.release());
  }

void WebGapi::readPixels(Device*, Pixmap& out, const PTexture, TextureFormat, uint32_t w, uint32_t h, uint32_t, bool) {
  // synchronous GPU readback is impossible on the browser main thread
  static bool warned = false;
  if(!warned) {
    warned = true;
    Log::e("[WebGPU] readPixels is not supported; returning a blank image");
    }
  out = Pixmap(std::max(1u, w), std::max(1u, h), TextureFormat::RGBA8);
  }

void WebGapi::readBytes(Device*, Buffer* buf, void* out, size_t size) {
  if(buf!=nullptr)
    buf->read(out, 0, size);
  }

AbstractGraphicsApi::CommandBuffer* WebGapi::createCommandBuffer(Device*) {
  return new WCommandBuffer();
  }

void WebGapi::present(Device*, Swapchain* sw) {
  ++g.frame;
  if(g.frame<=3 || g.frame%300==0) {
    Log::i("[WebGPU] frame ", g.frame, ": passes=", g.st.passes, " screen=", g.st.screenPasses,
           " draws=", g.st.draws, " skipped=", g.st.skipped, " dispatch=", g.st.dispatches,
           " submits=", g.st.submits, " errors=", g.errors);
    }
  g.st = Ctx::Stats();
  g.dbg = EM_ASM_INT({ window.__ogFrame = $0; return (typeof window.__OG_GAPI_DEBUG === 'number') ? window.__OG_GAPI_DEBUG : 0; }, g.frame);
  static_cast<WSwapchain*>(sw)->presented();
  for(auto t:g.deadTex) {
    wgpuTextureDestroy(t);
    wgpuTextureRelease(t);
    }
  g.deadTex.clear();
  for(auto b:g.deadBuf) {
    wgpuBufferDestroy(b);
    wgpuBufferRelease(b);
    }
  g.deadBuf.clear();
  }

std::shared_ptr<AbstractGraphicsApi::Fence> WebGapi::submit(Device*, CommandBuffer* cmd) {
  static_cast<WCommandBuffer*>(cmd)->submitTo(g.queue);
  return std::make_shared<WFence>();
  }

void WebGapi::getCaps(Device*, Props& props) {
  props.type = DeviceType::Discrete;
  std::snprintf(props.name, sizeof(props.name), "WebGPU");

  const auto& l = g.limits.limits;
  props.indirect.firstVertex   = true;
  props.indirect.firstInstance = g.firstInstance;

  props.vbo.maxAttribs = std::min<size_t>(16, l.maxVertexAttributes);
  props.vbo.maxStride  = l.maxVertexBufferArrayStride;
  props.ibo.maxValue   = 0xFFFFFFFF;

  props.ssbo.offsetAlign = l.minStorageBufferOffsetAlignment;
  props.ssbo.maxRange    = size_t(l.maxStorageBufferBindingSize);
  props.ubo.offsetAlign  = l.minUniformBufferOffsetAlignment;
  props.ubo.maxRange     = size_t(l.maxUniformBufferBindingSize);
  props.push.maxRange    = 128;

  props.render.maxColorAttachments  = l.maxColorAttachments;
  props.render.maxClipCullDistances = 0;
  props.render.maxViewportSize      = {int(l.maxTextureDimension2D), int(l.maxTextureDimension2D)};

  props.compute.maxGroups      = {int(l.maxComputeWorkgroupsPerDimension), int(l.maxComputeWorkgroupsPerDimension), int(l.maxComputeWorkgroupsPerDimension)};
  props.compute.maxGroupSize   = {int(l.maxComputeWorkgroupSizeX), int(l.maxComputeWorkgroupSizeY), int(l.maxComputeWorkgroupSizeZ)};
  props.compute.maxInvocations = int(l.maxComputeInvocationsPerWorkgroup);
  props.compute.maxSharedMemory = l.maxComputeWorkgroupStorageSize;

  props.tex2d.maxSize = l.maxTextureDimension2D;
  props.tex3d.maxSize = l.maxTextureDimension3D;

  props.descriptors.nonUniformIndexing = false;
  props.raytracing.rayQuery   = false;
  props.meshlets.taskShader   = false;
  props.meshlets.meshShader   = false;
  props.anisotropy            = true;
  props.maxAnisotropy         = 16;
  props.tesselationShader     = false;
  props.geometryShader        = false;
  props.storeAndAtomicVs      = false;
  props.storeAndAtomicFs      = false;
  props.memoryModel           = false;

  auto bit = [](TextureFormat f) { return uint64_t(1)<<uint64_t(f); };
  uint64_t smp = bit(R8) | bit(RG8) | bit(RGBA8) | bit(R16) | bit(RG16) | bit(RGBA16) |
                 bit(R32F) | bit(RG32F) | bit(RGBA32F) | bit(R32U) | bit(RG32U) | bit(RGBA32U) |
                 bit(Depth16) | bit(Depth24x8) | bit(Depth24S8) | bit(Depth32F) |
                 bit(R11G11B10UF) | bit(RGBA16F);
  if(g.bc)
    smp |= bit(DXT1) | bit(DXT3) | bit(DXT5);
  uint64_t att = bit(R8) | bit(RG8) | bit(RGBA8) | bit(R16) | bit(RG16) | bit(RGBA16) |
                 bit(R32F) | bit(RG32F) | bit(RGBA32F) | bit(R32U) | bit(RG32U) | bit(RGBA32U) | bit(RGBA16F);
  if(g.rg11b10render)
    att |= bit(R11G11B10UF);
  uint64_t dep = bit(Depth16) | bit(Depth24x8) | bit(Depth24S8) | bit(Depth32F);
  uint64_t sto = bit(RGBA8) | bit(R16) | bit(RG16) | bit(RGBA16) | bit(R32F) | bit(RG32F) | bit(RGBA32F) |
                 bit(R32U) | bit(RG32U) | bit(RGBA32U) | bit(RGBA16F);
  props.setSamplerFormats(smp);
  props.setAttachFormats(att);
  props.setDepthFormats(dep);
  props.setStorageFormats(sto);
  props.setAtomicFormats(0);
  }

#endif // __EMSCRIPTEN__
