#pragma once

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/html5.h>
#include <algorithm>
#include <cmath>

namespace Tempest {

// Device pixels per css pixel that the canvas drawing buffer (the swapchain) uses.
// Phones have a ratio of 2.5-3.5: in landscape the buffer gets to 2.5 Mpixels, and
// everything drawn at the swapchain size (UI, the final pass, the browser's
// compositing) costs more than the 3D scene at a low render scale.
// window.__ogPixelCap (the page's "screen sharpness"): 0 native, > 0 an upper
// limit of the ratio, anything else (default) automatic: the buffer is kept to
// about one megapixel, never below one pixel per css pixel.
inline double webPixelRatio(double cssW, double cssH) {
  const double dpr = emscripten_get_device_pixel_ratio();
  const double cap = EM_ASM_DOUBLE({ var c = window.__ogPixelCap; return (typeof c === 'number') ? c : -1; });
  if(cap==0)
    return dpr;
  if(cap>0)
    return std::min(dpr, cap);
  const double area = std::max(1.0, cssW*cssH);
  return std::min(dpr, std::max(1.0, std::sqrt(1.0e6/area)));
  }

}
#endif
