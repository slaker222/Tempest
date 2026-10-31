#pragma once

// BC1/BC2/BC3 (DXT1/3/5) blocks -> ASTC 4x4 LDR blocks, for devices that have no BC compression
// (phones): the texture stays compressed in video memory (16 bytes per 4x4 block instead of 64
// for RGBA8).
//
// The bit layout is the single partition, single plane ASTC block with 8-bit endpoints:
//   2-bit weights  (block mode 0x42): 4-colour BC1 blocks map almost exactly (weights 0, 21/64,
//                                      43/64, 1 against 0, 1/3, 2/3, 1)
//   3-bit weights  (block mode 0x53): 3-colour BC1 blocks (the middle colour is at 1/2)
// Blocks with alpha that varies inside the block are fitted by a line in RGBA space.

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace Tempest { namespace BcAstc {

struct Rgba { int c[4]; };

inline Rgba expand565(uint32_t v) {
  const int r = int((v>>11)&31), g = int((v>>5)&63), b = int(v&31);
  return {{ (r<<3)|(r>>2), (g<<2)|(g>>4), (b<<3)|(b>>2), 255 }};
  }

// a 128-bit block written bit by bit
struct BitBlock {
  uint8_t d[16] = {};
  void put(uint32_t pos, uint32_t value, uint32_t bits) {
    for(uint32_t i=0; i<bits; ++i)
      if((value>>i)&1u)
        d[(pos+i)>>3] |= uint8_t(1u<<((pos+i)&7));
    }
  // weight bit k (the first weight, least significant bit first) goes to block bit 127-k
  void putWeight(uint32_t bitIndex, uint32_t value, uint32_t bits) {
    for(uint32_t i=0; i<bits; ++i)
      if((value>>i)&1u) {
        const uint32_t p = 127u-(bitIndex+i);
        d[p>>3] |= uint8_t(1u<<(p&7));
        }
    }
  };

// endpoints e0/e1 (alpha used when withAlpha), weights per texel in raster order
inline void writeBlock(uint8_t* out, const Rgba& e0, const Rgba& e1, bool withAlpha,
                       const uint8_t w[16], uint32_t wBits) {
  BitBlock b;
  const uint32_t mode = (wBits==2) ? 0x42u : 0x53u;
  b.put(0, mode, 11);           // block mode, 4x4 grid; bits 11..12 (partitions-1) stay 0
  b.put(13, withAlpha ? 12u : 8u, 4); // colour endpoint mode: RGBA direct / RGB direct
  uint32_t pos = 17;
  for(int ch=0; ch<3; ++ch) {
    b.put(pos, uint32_t(e0.c[ch]), 8); pos += 8;
    b.put(pos, uint32_t(e1.c[ch]), 8); pos += 8;
    }
  if(withAlpha) {
    b.put(pos, uint32_t(e0.c[3]), 8); pos += 8;
    b.put(pos, uint32_t(e1.c[3]), 8); pos += 8;
    }
  for(uint32_t i=0; i<16; ++i)
    b.putWeight(i*wBits, w[i], wBits);
  std::memcpy(out, b.d, 16);
  }

inline int rgbSum(const Rgba& e) { return e.c[0]+e.c[1]+e.c[2]; }

// the decoder swaps and "blue contracts" the endpoints when the second one is darker: keep it from happening
inline void orderEndpoints(Rgba& e0, Rgba& e1, uint8_t w[16], uint32_t wBits) {
  if(rgbSum(e1) >= rgbSum(e0))
    return;
  std::swap(e0, e1);
  const uint8_t top = uint8_t((1u<<wBits)-1u);
  for(uint32_t i=0; i<16; ++i)
    w[i] = uint8_t(top-w[i]);
  }

// 8-byte BC1 colour block; alphaIsConst: texel alpha is constantAlpha for all texels (BC2/BC3 blocks
// with a constant alpha, opaque BC1)
inline void colorBlock(uint8_t* out, const uint8_t* bc, bool withAlpha, int alpha, bool four=false) {
  const uint32_t c0 = bc[0] | (uint32_t(bc[1])<<8), c1 = bc[2] | (uint32_t(bc[3])<<8);
  const uint32_t idx = bc[4] | (uint32_t(bc[5])<<8) | (uint32_t(bc[6])<<16) | (uint32_t(bc[7])<<24);
  Rgba e0 = expand565(c0), e1 = expand565(c1);
  e0.c[3] = e1.c[3] = alpha;
  uint8_t w[16];
  uint32_t wBits;
  if(four || c0>c1) {
    // 4 colours: index 0 = c0, 1 = c1, 2 = 2/3 c0 + 1/3 c1, 3 = 1/3 c0 + 2/3 c1
    static const uint8_t map4[4] = {0, 3, 1, 2};
    for(uint32_t i=0; i<16; ++i)
      w[i] = map4[(idx>>(2*i))&3];
    wBits = 2;
    } else {
    // 3 colours: 0 = c0, 1 = c1, 2 = (c0+c1)/2 (3 would be transparent, handled elsewhere)
    static const uint8_t map3[4] = {0, 7, 4, 4};
    for(uint32_t i=0; i<16; ++i)
      w[i] = map3[(idx>>(2*i))&3];
    wBits = 3;
    }
  orderEndpoints(e0, e1, w, wBits);
  writeBlock(out, e0, e1, withAlpha, w, wBits);
  }

// BC2/BC3 colour blocks are always in the 4-colour mode
inline void decodeColors(const uint8_t* bc, Rgba pal[4], bool four=false) {
  const uint32_t c0 = bc[0] | (uint32_t(bc[1])<<8), c1 = bc[2] | (uint32_t(bc[3])<<8);
  const Rgba e0 = expand565(c0), e1 = expand565(c1);
  pal[0] = e0;
  pal[1] = e1;
  for(int k=0; k<3; ++k) {
    if(four || c0>c1) {
      pal[2].c[k] = (2*e0.c[k]+e1.c[k])/3;
      pal[3].c[k] = (e0.c[k]+2*e1.c[k])/3;
      } else {
      pal[2].c[k] = (e0.c[k]+e1.c[k])/2;
      pal[3].c[k] = 0;
      }
    }
  pal[2].c[3] = 255;
  pal[3].c[3] = (four || c0>c1) ? 255 : 0;
  }

// blocks that had to be fitted (alpha that varies inside the block): not exact, see transcode()
inline size_t& lossyBlocks() {
  static thread_local size_t n = 0;
  return n;
  }

// fit a line through 16 RGBA texels: endpoints by the principal axis, then least squares refinement
inline void fitBlock(uint8_t* out, const Rgba px[16]) {
  ++lossyBlocks();
  float mean[4] = {};
  for(int i=0; i<16; ++i)
    for(int k=0; k<4; ++k)
      mean[k] += float(px[i].c[k]);
  for(int k=0; k<4; ++k)
    mean[k] /= 16.f;

  float cov[4][4] = {};
  for(int i=0; i<16; ++i) {
    float d[4];
    for(int k=0; k<4; ++k)
      d[k] = float(px[i].c[k])-mean[k];
    for(int a=0; a<4; ++a)
      for(int b=0; b<4; ++b)
        cov[a][b] += d[a]*d[b];
    }
  float axis[4] = {1, 1, 1, 1};
  for(int it=0; it<8; ++it) {
    float nx[4] = {};
    for(int a=0; a<4; ++a)
      for(int b=0; b<4; ++b)
        nx[a] += cov[a][b]*axis[b];
    float len = 0;
    for(int a=0; a<4; ++a)
      len += nx[a]*nx[a];
    len = std::sqrt(len);
    if(len<1e-6f)
      break;
    for(int a=0; a<4; ++a)
      axis[a] = nx[a]/len;
    }

  float tMin = 1e30f, tMax = -1e30f;
  for(int i=0; i<16; ++i) {
    float t = 0;
    for(int k=0; k<4; ++k)
      t += (float(px[i].c[k])-mean[k])*axis[k];
    tMin = std::min(tMin, t);
    tMax = std::max(tMax, t);
    }

  float p0[4], p1[4];
  for(int k=0; k<4; ++k) {
    p0[k] = mean[k]+axis[k]*tMin;
    p1[k] = mean[k]+axis[k]*tMax;
    }

  static const float wf[4] = {0.f, 21.f/64.f, 43.f/64.f, 1.f};
  uint8_t w[16] = {};
  for(int pass=0; pass<3; ++pass) {
    float dir[4], len2 = 0;
    for(int k=0; k<4; ++k) {
      dir[k] = p1[k]-p0[k];
      len2  += dir[k]*dir[k];
      }
    for(int i=0; i<16; ++i) {
      float t = 0;
      if(len2>1e-6f)
        for(int k=0; k<4; ++k)
          t += (float(px[i].c[k])-p0[k])*dir[k];
      t = (len2>1e-6f) ? t/len2 : 0.f;
      int   best = 0;
      float bd   = 1e30f;
      for(int j=0; j<4; ++j) {
        const float d = std::fabs(t-wf[j]);
        if(d<bd) { bd = d; best = j; }
        }
      w[i] = uint8_t(best);
      }
    // least squares for both endpoints with the chosen weights
    float aa = 0, ab = 0, bb = 0, ay[4] = {}, by[4] = {};
    for(int i=0; i<16; ++i) {
      const float b1 = wf[w[i]], a1 = 1.f-b1;
      aa += a1*a1; ab += a1*b1; bb += b1*b1;
      for(int k=0; k<4; ++k) {
        ay[k] += a1*float(px[i].c[k]);
        by[k] += b1*float(px[i].c[k]);
        }
      }
    const float det = aa*bb-ab*ab;
    if(std::fabs(det)<1e-6f)
      break;
    for(int k=0; k<4; ++k) {
      p0[k] = (ay[k]*bb-by[k]*ab)/det;
      p1[k] = (by[k]*aa-ay[k]*ab)/det;
      }
    }

  Rgba e0, e1;
  for(int k=0; k<4; ++k) {
    e0.c[k] = std::clamp(int(std::lround(p0[k])), 0, 255);
    e1.c[k] = std::clamp(int(std::lround(p1[k])), 0, 255);
    }
  orderEndpoints(e0, e1, w, 2);
  writeBlock(out, e0, e1, true, w, 2);
  }

// BC1: 8 bytes in, 16 bytes out
inline void bc1Block(uint8_t* out, const uint8_t* bc) {
  const uint32_t c0 = bc[0] | (uint32_t(bc[1])<<8), c1 = bc[2] | (uint32_t(bc[3])<<8);
  const uint32_t idx = bc[4] | (uint32_t(bc[5])<<8) | (uint32_t(bc[6])<<16) | (uint32_t(bc[7])<<24);
  if(c0<=c1) {
    bool transparent = false;
    for(uint32_t i=0; i<16; ++i)
      if(((idx>>(2*i))&3)==3)
        transparent = true;
    if(transparent) {
      Rgba pal[4];
      decodeColors(bc, pal);
      Rgba px[16];
      for(uint32_t i=0; i<16; ++i)
        px[i] = pal[(idx>>(2*i))&3];
      fitBlock(out, px);
      return;
      }
    }
  colorBlock(out, bc, false, 255);
  }

// BC2: 8 bytes of 4-bit alpha, then a BC1 colour block (always 4-colour mode)
inline void bc2Block(uint8_t* out, const uint8_t* bc) {
  int a[16];
  bool constAlpha = true;
  for(int i=0; i<16; ++i) {
    const int v = (bc[i>>1]>>((i&1)*4))&15;
    a[i] = v*17;
    constAlpha = constAlpha && a[i]==a[0];
    }
  if(constAlpha) {
    colorBlock(out, bc+8, a[0]!=255, a[0], true);
    return;
    }
  Rgba pal[4];
  decodeColors(bc+8, pal, true);
  const uint32_t idx = bc[12] | (uint32_t(bc[13])<<8) | (uint32_t(bc[14])<<16) | (uint32_t(bc[15])<<24);
  Rgba px[16];
  for(uint32_t i=0; i<16; ++i) {
    px[i] = pal[(idx>>(2*i))&3];
    px[i].c[3] = a[i];
    }
  fitBlock(out, px);
  }

// BC3: two alpha endpoints and 3-bit indices, then a BC1 colour block
inline void bc3Block(uint8_t* out, const uint8_t* bc) {
  int al[8];
  al[0] = bc[0];
  al[1] = bc[1];
  if(al[0]>al[1]) {
    for(int k=1; k<7; ++k)
      al[k+1] = ((7-k)*al[0]+k*al[1])/7;
    } else {
    for(int k=1; k<5; ++k)
      al[k+1] = ((5-k)*al[0]+k*al[1])/5;
    al[6] = 0;
    al[7] = 255;
    }
  uint64_t bits = 0;
  for(int i=0; i<6; ++i)
    bits |= uint64_t(bc[2+i])<<(8*i);
  int a[16];
  bool constAlpha = true;
  for(int i=0; i<16; ++i) {
    a[i] = al[(bits>>(3*i))&7];
    constAlpha = constAlpha && a[i]==a[0];
    }
  if(constAlpha) {
    colorBlock(out, bc+8, a[0]!=255, a[0], true);
    return;
    }
  Rgba pal[4];
  decodeColors(bc+8, pal, true);
  const uint32_t idx = bc[12] | (uint32_t(bc[13])<<8) | (uint32_t(bc[14])<<16) | (uint32_t(bc[15])<<24);
  Rgba px[16];
  for(uint32_t i=0; i<16; ++i) {
    px[i] = pal[(idx>>(2*i))&3];
    px[i].c[3] = a[i];
    }
  fitBlock(out, px);
  }

// kind: 1, 3 or 5 (DXT1/3/5). Returns the number of blocks that are not a (nearly) exact copy:
// those with alpha that varies inside the block, and 3-colour BC1 blocks with a transparent texel
inline size_t transcode(uint8_t* dst, const uint8_t* src, size_t blocks, int kind) {
  const size_t sb   = (kind==1) ? 8 : 16;
  const size_t was  = lossyBlocks();
  for(size_t i=0; i<blocks; ++i) {
    switch(kind) {
      case 1:  bc1Block(dst+i*16, src+i*sb); break;
      case 3:  bc2Block(dst+i*16, src+i*sb); break;
      default: bc3Block(dst+i*16, src+i*sb); break;
      }
    }
  return lossyBlocks()-was;
  }

// one block to 16 RGBA8 texels (raster order), the way a BC capable gpu decodes it
inline void decodeBlock(uint8_t* rgba, const uint8_t* bc, int kind) {
  const uint8_t* col = (kind==1) ? bc : bc+8;
  Rgba pal[4];
  decodeColors(col, pal, kind!=1);
  const uint32_t idx = col[4] | (uint32_t(col[5])<<8) | (uint32_t(col[6])<<16) | (uint32_t(col[7])<<24);
  for(uint32_t i=0; i<16; ++i) {
    const Rgba& p = pal[(idx>>(2*i))&3];
    for(int k=0; k<4; ++k)
      rgba[i*4+k] = uint8_t(p.c[k]);
    }
  if(kind==3) {
    for(int i=0; i<16; ++i)
      rgba[i*4+3] = uint8_t(((bc[i>>1]>>((i&1)*4))&15)*17);
    }
  if(kind==5) {
    int al[8];
    al[0] = bc[0];
    al[1] = bc[1];
    if(al[0]>al[1]) {
      for(int k=1; k<7; ++k)
        al[k+1] = ((7-k)*al[0]+k*al[1])/7;
      } else {
      for(int k=1; k<5; ++k)
        al[k+1] = ((5-k)*al[0]+k*al[1])/5;
      al[6] = 0;
      al[7] = 255;
      }
    uint64_t bits = 0;
    for(int i=0; i<6; ++i)
      bits |= uint64_t(bc[2+i])<<(8*i);
    for(int i=0; i<16; ++i)
      rgba[i*4+3] = uint8_t(al[(bits>>(3*i))&7]);
    }
  }

}}
