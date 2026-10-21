#if defined(TEMPEST_BUILD_VULKAN)

#include "vshader.h"

#include "vdevice.h"

using namespace Tempest::Detail;

static uint32_t fnv1a32(const void* data, size_t size) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint32_t hash = 2166136261u;
  for(size_t i=0; i<size; ++i) {
    hash ^= p[i];
    hash *= 16777619u;
  }
  return hash;
}

VShader::VShader(VDevice& device, const void *source, size_t src_size)
  :Shader(source, src_size), device(device.device.impl), srcSize(src_size), spirvHash(fnv1a32(source, src_size)) {
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = src_size;
  createInfo.pCode    = reinterpret_cast<const uint32_t*>(source);

  if(vkCreateShaderModule(device.device.impl,&createInfo,nullptr,&impl)!=VK_SUCCESS)
    throw std::system_error(Tempest::GraphicsErrc::InvalidShaderModule, "\"" + dbg.source + "\"");

  VkDebugMarkerObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT};
  nameInfo.objectType   = VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT;
  nameInfo.object       = impl;
  nameInfo.pObjectName  = this->dbgShortName();
  device.vkDebugMarkerSetObjectName(device.device.impl, &nameInfo);
  }

VShader::VShader(VDevice& device)
  :Shader(), device(device.device.impl) {
  }

VShader::~VShader() {
  if(impl!=VK_NULL_HANDLE)
    vkDestroyShaderModule(device,impl,nullptr);
  }

#endif
