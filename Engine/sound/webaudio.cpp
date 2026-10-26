#if defined(__EMSCRIPTEN__) && !defined(TEMPEST_BUILD_AUDIO)

// Web Audio implementation of SoundDevice / SoundEffect.
// Static sounds become cached AudioBuffers played through Gain+Panner nodes
// (linear distance model, like the OpenAL backend). Streaming producers are
// pumped from a timer on the main thread and scheduled as consecutive chunks.

#include "sounddevice.h"
#include "soundeffect.h"
#include "sound.h"

#include <Tempest/Except>
#include <Tempest/Log>

#include <emscripten.h>
#include <emscripten/eventloop.h>
#if defined(__EMSCRIPTEN_PTHREADS__)
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace Tempest;

EM_JS(void, oga_init, (), {
  if (Module.oga) return;
  const A = Module.oga = { ctx: null, master: null, bufs: [null], srcs: [null] };
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  A.ctx    = new AC({ latencyHint: 'interactive' });
  A.master = A.ctx.createGain();
  A.master.connect(A.ctx.destination);
  const resume = () => { if (A.ctx.state !== 'running') A.ctx.resume(); };
  for (const e of ['pointerdown', 'keydown', 'touchstart'])
    window.addEventListener(e, resume, true);
  resume();
});

EM_JS(int, oga_buffer, (const void* ptr, int bytes, int channels, int bits, int rate), {
  const A = Module.oga;
  if (!A || !A.ctx || channels < 1) return 0;
  const frames = Math.floor(bytes / (channels * (bits >> 3)));
  if (frames <= 0 || rate <= 0) return 0;
  const buf = A.ctx.createBuffer(channels, frames, rate);
  for (let ch = 0; ch < channels; ++ch) {
    const d = buf.getChannelData(ch);
    if (bits === 16) {
      const base = ptr >> 1;
      for (let i = 0; i < frames; ++i) d[i] = HEAP16[base + i * channels + ch] / 32768;
    } else {
      for (let i = 0; i < frames; ++i) d[i] = (HEAPU8[ptr + i * channels + ch] - 128) / 128;
    }
  }
  A.bufs.push(buf);
  return A.bufs.length - 1;
});

EM_JS(void, oga_buffer_free, (int id), {
  const A = Module.oga;
  if (A && id > 0) A.bufs[id] = null;
});

EM_JS(int, oga_source, (int buf, int channels, int rate), {
  const A = Module.oga;
  if (!A || !A.ctx) return 0;
  const s = { buf: A.bufs[buf] || null, channels: channels, rate: rate,
              gain: 1, pos: [0, 0, 0], maxDist: 1e30,
              node: null, playing: false, started: false, ended: false,
              offset: 0, startedAt: 0, chunks: [], next: 0 };
  s.gainNode = A.ctx.createGain();
  s.panner   = A.ctx.createPanner();
  s.panner.panningModel  = 'equalpower';
  s.panner.distanceModel = 'linear';
  s.panner.refDistance   = 0;
  s.panner.maxDistance   = 1e30;
  s.panner.rolloffFactor = 1;
  s.gainNode.connect(s.panner);
  s.panner.connect(A.master);
  A.srcs.push(s);
  return A.srcs.length - 1;
});

EM_JS(void, oga_source_free, (int id), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s) return;
  if (s.node) { s.node.onended = null; try { s.node.stop(); } catch (e) {} }
  for (const c of s.chunks) { try { c.stop(); } catch (e) {} }
  s.gainNode.disconnect();
  s.panner.disconnect();
  A.srcs[id] = null;
});

EM_JS(void, oga_play, (int id), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s || s.playing) return;
  s.playing   = true;
  s.started   = true;
  s.ended     = false;
  s.startedAt = A.ctx.currentTime;
  if (!s.buf) { s.next = 0; return; } // streaming source: chunks are pushed by the pump
  if (s.offset >= s.buf.duration) s.offset = 0;
  const n = A.ctx.createBufferSource();
  n.buffer = s.buf;
  n.connect(s.gainNode);
  n.onended = () => { if (s.node === n) { s.node = null; s.playing = false; s.ended = true; s.offset = 0; } };
  s.node = n;
  n.start(0, s.offset);
});

EM_JS(void, oga_pause, (int id), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s || !s.playing) return;
  s.offset += A.ctx.currentTime - s.startedAt;
  s.playing = false;
  if (s.node) { const n = s.node; s.node = null; n.onended = null; try { n.stop(); } catch (e) {} }
  for (const c of s.chunks) { try { c.stop(); } catch (e) {} }
  s.chunks = [];
});

// 0 = initial/stopped, 1 = playing, 2 = paused
EM_JS(int, oga_state, (int id), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s) return 0;
  if (s.playing) return 1;
  return (s.started && !s.ended) ? 2 : 0;
});

EM_JS(double, oga_time, (int id), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s) return 0;
  return s.playing ? s.offset + (A.ctx.currentTime - s.startedAt) : s.offset;
});

EM_JS(void, oga_set, (int id, int what, float x, float y, float z), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s) return;
  const t = A.ctx.currentTime;
  // every setValueAtTime adds an automation event: skip unchanged values
  if (what === 0) {
    const p = s.pos;
    if (p && p[0] === x && p[1] === y && p[2] === z) return;
    s.pos = [x, y, z];
    s.panner.positionX.setValueAtTime(x, t);
    s.panner.positionY.setValueAtTime(y, t);
    s.panner.positionZ.setValueAtTime(z, t);
  } else if (what === 1) {
    const d = Math.max(x, 1e-3);
    if (s.maxDist === d) return;
    s.maxDist = d;
    s.panner.maxDistance = d;
  } else if (what === 2) {
    if (s.gain === x && s.gainSet) return;
    s.gain = x;
    s.gainSet = true;
    s.gainNode.gain.setValueAtTime(x, t);
  }
});

EM_JS(float, oga_get, (int id, int what), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s) return 0;
  if (what < 3) return s.pos[what];
  return s.gain;
});

// frames still needed to keep `ahead` seconds queued on a playing stream
EM_JS(int, oga_stream_need, (int id, float ahead), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s || !s.playing || A.ctx.state !== 'running') return 0;
  const now = A.ctx.currentTime;
  if (s.next < now + 0.02) s.next = now + 0.02;
  const left = (now + ahead) - s.next;
  return left > 0 ? Math.ceil(left * s.rate) : 0;
});

EM_JS(void, oga_stream_push, (int id, const int16_t* ptr, int frames), {
  const A = Module.oga;
  const s = A && A.srcs[id];
  if (!s || frames <= 0) return;
  const ch  = s.channels;
  const buf = A.ctx.createBuffer(ch, frames, s.rate);
  const base = ptr >> 1;
  for (let c = 0; c < ch; ++c) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < frames; ++i) d[i] = HEAP16[base + i * ch + c] / 32768;
  }
  const n = A.ctx.createBufferSource();
  n.buffer = buf;
  n.connect(s.gainNode);
  n.onended = () => { const k = s.chunks.indexOf(n); if (k >= 0) s.chunks.splice(k, 1); };
  n.start(s.next);
  s.next += frames / s.rate;
  s.chunks.push(n);
});

EM_JS(void, oga_listener, (int what, float a, float b, float c, float d, float e, float f), {
  const A = Module.oga;
  if (!A || !A.ctx) return;
  const L = A.ctx.listener, t = A.ctx.currentTime;
  const k = what === 2 ? [a] : what === 0 ? [a, b, c] : [a, b, c, d, e, f];
  const last = (A.lst || (A.lst = []))[what];
  if (last && last.length === k.length && last.every((v, i) => v === k[i])) return;
  A.lst[what] = k;
  if (what === 0) {
    if (L.positionX) { L.positionX.setValueAtTime(a, t); L.positionY.setValueAtTime(b, t); L.positionZ.setValueAtTime(c, t); }
    else L.setPosition(a, b, c);
  } else if (what === 1) {
    if (L.forwardX) {
      L.forwardX.setValueAtTime(a, t); L.forwardY.setValueAtTime(b, t); L.forwardZ.setValueAtTime(c, t);
      L.upX.setValueAtTime(d, t);      L.upY.setValueAtTime(e, t);      L.upZ.setValueAtTime(f, t);
    } else L.setOrientation(a, b, c, d, e, f);
  } else if (what === 2) {
    A.master.gain.setValueAtTime(a, t);
  }
});

namespace {

// The AudioContext and the tables below live on the main thread; sounds are
// also started and moved from worker threads (npc animation runs in parallel),
// those calls run synchronously on the main thread.
template<class F>
auto onMain(F&& f) -> decltype(f()) {
#if defined(__EMSCRIPTEN_PTHREADS__)
  if(!emscripten_is_main_runtime_thread()) {
    using R = decltype(f());
    if constexpr(std::is_void_v<R>) {
      auto call = [](void* p) { (*static_cast<F*>(p))(); };
      emscripten_proxy_sync(emscripten_proxy_get_system_queue(), emscripten_main_runtime_thread_id(), call, &f);
      return;
      } else {
      struct Ctx { F& f; R r; } ctx{f, R()};
      auto call = [](void* p) { auto& c = *static_cast<Ctx*>(p); c.r = c.f(); };
      emscripten_proxy_sync(emscripten_proxy_get_system_queue(), emscripten_main_runtime_thread_id(), call, &ctx);
      return ctx.r;
      }
    }
#endif
  return f();
  }

std::unordered_map<const void*, int> g_buffers; // Sound::Data -> AudioBuffer id

struct Stream {
  virtual ~Stream() = default;
  virtual void pump() = 0;
  virtual void prefill() {}
  };
std::vector<Stream*> g_streams;
bool                 g_pumpStarted = false;

#if defined(__EMSCRIPTEN_PTHREADS__)
// Streams (music) are rendered ahead on an audio thread, as OpenAL does on
// native builds; the main thread only hands the samples to Web Audio. The
// main thread never blocks on this lock: the audio thread may be waiting for
// a call proxied to the main thread.
std::mutex g_audioMx;

void lockAudioOnMain() {
  while(!g_audioMx.try_lock())
    emscripten_proxy_execute_queue(emscripten_proxy_get_system_queue());
  }

void audioThread() {
  for(;;) {
    {
    std::lock_guard<std::mutex> guard(g_audioMx);
    for(size_t i=0; i<g_streams.size(); ++i)
      g_streams[i]->prefill();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
#endif

void pumpAll(void*) {
#if defined(__EMSCRIPTEN_PTHREADS__)
  std::unique_lock<std::mutex> lck(g_audioMx, std::try_to_lock);
  if(!lck.owns_lock())
    return; // Web Audio is queued 0.3 s ahead: the next pump catches up
#endif
  for(size_t i=0; i<g_streams.size(); ++i)
    g_streams[i]->pump();
  }

}

namespace Tempest {
namespace Detail {
void webAudioReleaseSound(const void* data) {
  onMain([data]() {
    auto it = g_buffers.find(data);
    if(it==g_buffers.end())
      return;
    oga_buffer_free(it->second);
    g_buffers.erase(it);
    });
  }
}
}

// ------------------------------------------------------------------ device

struct SoundDevice::Data {};

SoundDevice::SoundDevice() : data(new Data()) {
  oga_init();
  if(!g_pumpStarted) {
    g_pumpStarted = true;
    emscripten_set_interval(pumpAll, 40, nullptr);
#if defined(__EMSCRIPTEN_PTHREADS__)
    std::thread(audioThread).detach();
#endif
    }
  }

SoundDevice::SoundDevice(std::string_view) : SoundDevice() {}

SoundDevice::~SoundDevice() {}

std::vector<SoundDevice::Props> SoundDevice::devices() {
  std::vector<Props> ret(1);
  std::snprintf(ret[0].name, sizeof(ret[0].name), "Web Audio");
  return ret;
  }

SoundEffect SoundDevice::load(const char* fname) {
  Tempest::Sound s(fname);
  return load(s);
  }

SoundEffect SoundDevice::load(IDevice& d) {
  Tempest::Sound s(d);
  return load(s);
  }

SoundEffect SoundDevice::load(const Sound& snd) {
  return SoundEffect(*this, snd);
  }

SoundEffect SoundDevice::load(std::unique_ptr<SoundProducer>&& p) {
  return SoundEffect(*this, std::move(p));
  }

void SoundDevice::process() {}
void SoundDevice::suspend() {}

void SoundDevice::setListenerPosition(const Vec3& p) {
  onMain([&]() { oga_listener(0, p.x, p.y, p.z, 0, 0, 0); });
  }

void SoundDevice::setListenerPosition(float x, float y, float z) {
  onMain([&]() { oga_listener(0, x, y, z, 0, 0, 0); });
  }

void SoundDevice::setListenerDirection(const Vec3& f, const Vec3& up) {
  onMain([&]() { oga_listener(1, f.x, f.y, f.z, up.x, up.y, up.z); });
  }

void SoundDevice::setListenerDirection(float dx, float dy, float dz, float ux, float uy, float uz) {
  onMain([&]() { oga_listener(1, dx, dy, dz, ux, uy, uz); });
  }

void SoundDevice::setGlobalVolume(float v) {
  onMain([&]() { oga_listener(2, v, 0, 0, 0, 0, 0); });
  }

void* SoundDevice::context() {
  return nullptr;
  }

// ------------------------------------------------------------------ effect

SoundProducer::SoundProducer(uint16_t frequency, uint16_t channels)
  :frequency(frequency), channels(channels) {
  if(channels!=1 && channels!=2)
    throw std::system_error(Tempest::SoundErrc::InvalidChannelsCount);
  }

struct SoundEffect::Impl : Stream {
  int                            src = 0;
  std::shared_ptr<Sound::Data>   data;
  std::unique_ptr<SoundProducer> producer;
  std::vector<int16_t>           scratch;

  Impl() = default;

  Impl(const Sound& s) : data(s.data) {
    if(data==nullptr)
      return;
    onMain([this]() { init(); });
    }

  void init() {
    int buf = 0;
    auto it = g_buffers.find(data.get());
    if(it!=g_buffers.end()) {
      buf = it->second;
      } else {
      const int fmt      = data->format;
      const int channels = (fmt==0x1102 || fmt==0x1103) ? 2 : 1;  // AL_FORMAT_STEREO8/16
      const int bits     = (fmt==0x1100 || fmt==0x1102) ? 8 : 16;  // AL_FORMAT_MONO8/STEREO8
      buf = oga_buffer(data->ptr.get(), int(data->byteSize), channels, bits, int(data->frequency));
      if(buf!=0)
        g_buffers[data.get()] = buf;
      }
    if(buf!=0)
      src = oga_source(buf, 0, 0);
    }

  Impl(std::unique_ptr<SoundProducer>&& p) : producer(std::move(p)) {
    onMain([this]() {
      src = oga_source(0, producer->channels, producer->frequency);
      if(src==0)
        return;
#if defined(__EMSCRIPTEN_PTHREADS__)
      lockAudioOnMain();
      g_streams.push_back(this);
      g_audioMx.unlock();
#else
      g_streams.push_back(this);
#endif
      });
    }

  ~Impl() override {
    onMain([this]() { release(); });
    }

  void release() {
    if(producer!=nullptr) {
#if defined(__EMSCRIPTEN_PTHREADS__)
      lockAudioOnMain();
#endif
      auto it = std::find(g_streams.begin(), g_streams.end(), static_cast<Stream*>(this));
      if(it!=g_streams.end())
        g_streams.erase(it);
#if defined(__EMSCRIPTEN_PTHREADS__)
      g_audioMx.unlock();
#endif
      }
    if(src!=0)
      oga_source_free(src);
    }

  void pump() override {
    int need = oga_stream_need(src, 0.3f);
    while(need>0) {
      const int frames = std::min(need, 2048);
      // renderSound implementations may write up to 8 bytes per frame
      scratch.assign(size_t(frames)*4, 0);
      const int ready = takeAhead(scratch.data(), frames);
      if(ready<frames)
        producer->renderSound(scratch.data()+size_t(ready)*producer->channels, size_t(frames-ready));
      oga_stream_push(src, scratch.data(), frames);
      need -= frames;
      }
    }

  // samples rendered ahead (audio thread), aheadCnt frames from aheadRd on
  static constexpr int   AheadFrames = 4096, AheadChunk = 1024;
  std::vector<int16_t>   ahead, aheadTmp;
  int                    aheadRd = 0, aheadCnt = 0;

  void prefill() override {
#if defined(__EMSCRIPTEN_PTHREADS__)
    const int ch = producer->channels;
    if(ahead.empty())
      ahead.resize(size_t(AheadFrames)*ch);
    while(aheadCnt+AheadChunk<=AheadFrames) {
      aheadTmp.assign(size_t(AheadChunk)*4, 0);
      producer->renderSound(aheadTmp.data(), size_t(AheadChunk));
      for(int i=0; i<AheadChunk; ++i) {
        const int at = (aheadRd+aheadCnt+i)%AheadFrames;
        std::memcpy(&ahead[size_t(at)*ch], &aheadTmp[size_t(i)*ch], size_t(ch)*sizeof(int16_t));
        }
      aheadCnt += AheadChunk;
      }
#endif
    }

  int takeAhead(int16_t* dst, int frames) {
    const int ch = producer->channels;
    const int n  = std::min(frames, aheadCnt);
    for(int i=0; i<n; ++i) {
      std::memcpy(dst+size_t(i)*ch, &ahead[size_t(aheadRd)*ch], size_t(ch)*sizeof(int16_t));
      aheadRd = (aheadRd+1)%AheadFrames;
      }
    aheadCnt -= n;
    return n;
    }
  };

SoundEffect::SoundEffect(SoundDevice&, const Sound& src) : impl(new Impl(src)) {}

SoundEffect::SoundEffect(SoundDevice&, std::unique_ptr<SoundProducer>&& src) : impl(new Impl(std::move(src))) {}

SoundEffect::SoundEffect() : impl(new Impl()) {}

SoundEffect::SoundEffect(SoundEffect&& s) : impl(std::move(s.impl)) {}

SoundEffect::~SoundEffect() {}

SoundEffect& SoundEffect::operator=(SoundEffect&& s) {
  std::swap(impl, s.impl);
  return *this;
  }

void SoundEffect::play() {
  if(impl && impl->src!=0)
    onMain([this]() { oga_play(impl->src); });
  }

void SoundEffect::pause() {
  if(impl && impl->src!=0)
    onMain([this]() { oga_pause(impl->src); });
  }

bool SoundEffect::isEmpty() const {
  return impl==nullptr || impl->src==0;
  }

bool SoundEffect::isFinished() const {
  if(impl==nullptr || impl->src==0)
    return true;
  return onMain([this]() { return oga_state(impl->src); })==0;
  }

uint64_t SoundEffect::timeLength() const {
  if(impl && impl->data)
    return impl->data->timeLength();
  return 0;
  }

uint64_t SoundEffect::currentTime() const {
  if(impl==nullptr || impl->src==0)
    return 0;
  return uint64_t(onMain([this]() { return oga_time(impl->src); })*1000.0);
  }

void SoundEffect::setPosition(const Vec3& p) {
  setPosition(p.x, p.y, p.z);
  }

void SoundEffect::setPosition(float x, float y, float z) {
  if(impl && impl->src!=0)
    onMain([&]() { oga_set(impl->src, 0, x, y, z); });
  }

Vec3 SoundEffect::position() const {
  if(impl==nullptr || impl->src==0)
    return Vec3();
  return onMain([this]() { return Vec3(oga_get(impl->src, 0), oga_get(impl->src, 1), oga_get(impl->src, 2)); });
  }

float SoundEffect::x() const { return position().x; }
float SoundEffect::y() const { return position().y; }
float SoundEffect::z() const { return position().z; }

void SoundEffect::setMaxDistance(float dist) {
  if(impl && impl->src!=0)
    onMain([&]() { oga_set(impl->src, 1, dist, 0, 0); });
  }

void SoundEffect::setVolume(float val) {
  if(impl && impl->src!=0)
    onMain([&]() { oga_set(impl->src, 2, val, 0, 0); });
  }

float SoundEffect::volume() const {
  if(impl==nullptr || impl->src==0)
    return 0;
  return onMain([this]() { return oga_get(impl->src, 3); });
  }

#endif
