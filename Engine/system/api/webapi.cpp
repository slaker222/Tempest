#if defined(__EMSCRIPTEN__)

#include "webapi.h"

#include <Tempest/Event>
#include <Tempest/Window>
#include <Tempest/Log>

#include <cstring>
#include <cstdlib>
#include <algorithm>

#include <emscripten.h>
#include <emscripten/html5.h>

using namespace Tempest;

struct SystemApi::Window {
  Tempest::Window* owner      = nullptr;
  uint32_t         width      = 1280;
  uint32_t         height     = 720;
  bool             fullscreen = false;
  };

static SystemApi::Window* s_currentWindow = nullptr;
static WebApi::AppCallBack* s_appCallback = nullptr;
static bool s_isRunning = true;

WebApi::WebApi() {
  }

WebApi::~WebApi() {
  }

static Event::KeyType mapEmscriptenKey(const char* code) {
  if(!code) return Event::K_NoKey;
  if(std::strcmp(code, "KeyA") == 0) return Event::K_A;
  if(std::strcmp(code, "KeyB") == 0) return Event::K_B;
  if(std::strcmp(code, "KeyC") == 0) return Event::K_C;
  if(std::strcmp(code, "KeyD") == 0) return Event::K_D;
  if(std::strcmp(code, "KeyE") == 0) return Event::K_E;
  if(std::strcmp(code, "KeyF") == 0) return Event::K_F;
  if(std::strcmp(code, "KeyG") == 0) return Event::K_G;
  if(std::strcmp(code, "KeyH") == 0) return Event::K_H;
  if(std::strcmp(code, "KeyI") == 0) return Event::K_I;
  if(std::strcmp(code, "KeyJ") == 0) return Event::K_J;
  if(std::strcmp(code, "KeyK") == 0) return Event::K_K;
  if(std::strcmp(code, "KeyL") == 0) return Event::K_L;
  if(std::strcmp(code, "KeyM") == 0) return Event::K_M;
  if(std::strcmp(code, "KeyN") == 0) return Event::K_N;
  if(std::strcmp(code, "KeyO") == 0) return Event::K_O;
  if(std::strcmp(code, "KeyP") == 0) return Event::K_P;
  if(std::strcmp(code, "KeyQ") == 0) return Event::K_Q;
  if(std::strcmp(code, "KeyR") == 0) return Event::K_R;
  if(std::strcmp(code, "KeyS") == 0) return Event::K_S;
  if(std::strcmp(code, "KeyT") == 0) return Event::K_T;
  if(std::strcmp(code, "KeyU") == 0) return Event::K_U;
  if(std::strcmp(code, "KeyV") == 0) return Event::K_V;
  if(std::strcmp(code, "KeyW") == 0) return Event::K_W;
  if(std::strcmp(code, "KeyX") == 0) return Event::K_X;
  if(std::strcmp(code, "KeyY") == 0) return Event::K_Y;
  if(std::strcmp(code, "KeyZ") == 0) return Event::K_Z;
  if(std::strcmp(code, "Space") == 0) return Event::K_Space;
  if(std::strcmp(code, "Escape") == 0) return Event::K_ESCAPE;
  if(std::strcmp(code, "Enter") == 0) return Event::K_Return;
  if(std::strcmp(code, "Backspace") == 0) return Event::K_Back;
  if(std::strcmp(code, "Tab") == 0) return Event::K_Tab;
  if(std::strcmp(code, "ShiftLeft") == 0) return Event::K_LShift;
  if(std::strcmp(code, "ShiftRight") == 0) return Event::K_RShift;
  if(std::strcmp(code, "ControlLeft") == 0) return Event::K_LControl;
  if(std::strcmp(code, "ControlRight") == 0) return Event::K_RControl;
  if(std::strcmp(code, "AltLeft") == 0) return Event::K_LAlt;
  if(std::strcmp(code, "AltRight") == 0) return Event::K_RAlt;
  if(std::strcmp(code, "ArrowUp") == 0) return Event::K_Up;
  if(std::strcmp(code, "ArrowDown") == 0) return Event::K_Down;
  if(std::strcmp(code, "ArrowLeft") == 0) return Event::K_Left;
  if(std::strcmp(code, "ArrowRight") == 0) return Event::K_Right;
  if(std::strcmp(code, "F1") == 0) return Event::K_F1;
  if(std::strcmp(code, "F2") == 0) return Event::K_F2;
  if(std::strcmp(code, "F3") == 0) return Event::K_F3;
  if(std::strcmp(code, "F4") == 0) return Event::K_F4;
  if(std::strcmp(code, "F5") == 0) return Event::K_F5;
  if(std::strcmp(code, "F6") == 0) return Event::K_F6;
  if(std::strcmp(code, "F7") == 0) return Event::K_F7;
  if(std::strcmp(code, "F8") == 0) return Event::K_F8;
  if(std::strcmp(code, "F9") == 0) return Event::K_F9;
  if(std::strcmp(code, "F10") == 0) return Event::K_F10;
  if(std::strcmp(code, "F11") == 0) return Event::K_F11;
  if(std::strcmp(code, "F12") == 0) return Event::K_F12;
  return Event::K_NoKey;
}

static EM_BOOL onEmscriptenMouseDown(int, const EmscriptenMouseEvent* e, void* userData) {
  auto* w = static_cast<SystemApi::Window*>(userData);
  if(!w || !w->owner) return EM_FALSE;
  Event::MouseButton btn = Event::ButtonLeft;
  if(e->button == 1) btn = Event::ButtonMid;
  if(e->button == 2) btn = Event::ButtonRight;
  MouseEvent evt(int(e->targetX), int(e->targetY), btn, Event::M_NoModifier, 0, 0, Event::MouseDown);
  WebApi::dispatchMouseDown(*w->owner, evt);
  return EM_TRUE;
}

static EM_BOOL onEmscriptenMouseUp(int, const EmscriptenMouseEvent* e, void* userData) {
  auto* w = static_cast<SystemApi::Window*>(userData);
  if(!w || !w->owner) return EM_FALSE;
  Event::MouseButton btn = Event::ButtonLeft;
  if(e->button == 1) btn = Event::ButtonMid;
  if(e->button == 2) btn = Event::ButtonRight;
  MouseEvent evt(int(e->targetX), int(e->targetY), btn, Event::M_NoModifier, 0, 0, Event::MouseUp);
  WebApi::dispatchMouseUp(*w->owner, evt);
  return EM_TRUE;
}

static bool isPointerLocked() {
  EmscriptenPointerlockChangeEvent st = {};
  return emscripten_get_pointerlock_status(&st)==EMSCRIPTEN_RESULT_SUCCESS && st.isActive;
  }

static EM_BOOL onEmscriptenMouseMove(int, const EmscriptenMouseEvent* e, void* userData) {
  auto* w = static_cast<SystemApi::Window*>(userData);
  if(!w || !w->owner) return EM_FALSE;
  // under pointer lock the cursor position freezes: move a virtual cursor by
  // the reported motion instead, so consumers keep seeing position deltas
  static int vx = 0, vy = 0;
  if(isPointerLocked()) {
    vx += int(e->movementX);
    vy += int(e->movementY);
    } else {
    vx = int(e->targetX);
    vy = int(e->targetY);
    }
  const int x = vx, y = vy;
  MouseEvent evt(x, y, Event::ButtonNone, Event::M_NoModifier, 0, 0, Event::MouseMove);
  WebApi::dispatchMouseMove(*w->owner, evt);
  return EM_TRUE;
}

static EM_BOOL onEmscriptenWheel(int, const EmscriptenWheelEvent* e, void* userData) {
  auto* w = static_cast<SystemApi::Window*>(userData);
  if(!w || !w->owner) return EM_FALSE;
  int delta = (e->deltaY < 0) ? 120 : -120;
  MouseEvent evt(int(e->mouse.targetX), int(e->mouse.targetY), Event::ButtonNone, Event::M_NoModifier, delta, 0, Event::MouseWheel);
  WebApi::dispatchMouseWheel(*w->owner, evt);
  return EM_TRUE;
}

static EM_BOOL onEmscriptenKeyDown(int, const EmscriptenKeyboardEvent* e, void* userData) {
  auto* w = static_cast<SystemApi::Window*>(userData);
  if(!w || !w->owner) return EM_FALSE;
  auto k = mapEmscriptenKey(e->code);
  KeyEvent evt(k, uint32_t(e->keyCode), Event::M_NoModifier, Event::KeyDown);
  WebApi::dispatchKeyDown(*w->owner, evt, uint32_t(e->keyCode));
  return EM_TRUE;
}

static EM_BOOL onEmscriptenKeyUp(int, const EmscriptenKeyboardEvent* e, void* userData) {
  auto* w = static_cast<SystemApi::Window*>(userData);
  if(!w || !w->owner) return EM_FALSE;
  auto k = mapEmscriptenKey(e->code);
  KeyEvent evt(k, uint32_t(e->keyCode), Event::M_NoModifier, Event::KeyUp);
  WebApi::dispatchKeyUp(*w->owner, evt, uint32_t(e->keyCode));
  return EM_TRUE;
}

SystemApi::Window* WebApi::implCreateWindow(Tempest::Window* owner, uint32_t width, uint32_t height) {
  auto* w = new SystemApi::Window();
  w->owner  = owner;
  w->width  = (width > 0) ? width : 1280;
  w->height = (height > 0) ? height : 720;

  double cssW = 0, cssH = 0;
  if(emscripten_get_element_css_size("#canvas", &cssW, &cssH) >= 0 && cssW > 0 && cssH > 0) {
    w->width  = static_cast<uint32_t>(cssW);
    w->height = static_cast<uint32_t>(cssH);
    }

  s_currentWindow = w;

  emscripten_set_mousedown_callback("#canvas", w, 1, onEmscriptenMouseDown);
  emscripten_set_mouseup_callback("#canvas", w, 1, onEmscriptenMouseUp);
  emscripten_set_mousemove_callback("#canvas", w, 1, onEmscriptenMouseMove);
  emscripten_set_wheel_callback("#canvas", w, 1, onEmscriptenWheel);
  emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, w, 1, onEmscriptenKeyDown);
  emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, w, 1, onEmscriptenKeyUp);

  return w;
  }

SystemApi::Window* WebApi::implCreateWindow(Tempest::Window* owner, ShowMode sm) {
  uint32_t w = 1280, h = 720;
  if(sm == ShowMode::FullScreen) {
    w = 1920;
    h = 1080;
    }
  return implCreateWindow(owner, w, h);
  }

void WebApi::implDestroyWindow(SystemApi::Window* w) {
  emscripten_set_mousedown_callback("#canvas", nullptr, 0, nullptr);
  emscripten_set_mouseup_callback("#canvas", nullptr, 0, nullptr);
  emscripten_set_mousemove_callback("#canvas", nullptr, 0, nullptr);
  emscripten_set_wheel_callback("#canvas", nullptr, 0, nullptr);
  emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, 0, nullptr);
  emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, 0, nullptr);

  if(s_currentWindow == w) {
    s_currentWindow = nullptr;
    }
  delete w;
  }

void WebApi::implExit() {
  s_isRunning = false;
  }

Rect WebApi::implWindowClientRect(SystemApi::Window* w) {
  if(w == nullptr)
    return Rect(0, 0, 1280, 720);
  return Rect(0, 0, int(w->width), int(w->height));
  }

bool WebApi::implSetAsFullscreen(SystemApi::Window* w, bool fullScreen) {
  // The page owns fullscreen (its buttons put the whole player, controls
  // included, fullscreen or into window mode). A request for the bare canvas
  // is deferred by the browser to the next click and then crops the page
  // layout: only the flag is kept.
  if(w != nullptr)
    w->fullscreen = fullScreen;
  return true;
  }

bool WebApi::implIsFullscreen(SystemApi::Window* w) {
  // pointer lock enables mouse-look like a fullscreen window does natively
  return w ? (w->fullscreen || isPointerLocked()) : false;
  }

float WebApi::implUiScale(SystemApi::Window*) {
  return static_cast<float>(emscripten_get_device_pixel_ratio());
  }

void WebApi::implSetWindowTitle(SystemApi::Window*, const char* utf8) {
  if(utf8 != nullptr) {
    emscripten_set_window_title(utf8);
    }
  }

void WebApi::implSetCursorPosition(SystemApi::Window*, int, int) {
  }

void WebApi::implShowCursor(SystemApi::Window*, CursorShape) {
  }

bool WebApi::implIsRunning() {
  return s_isRunning;
  }

static void mainLoopIteration() {
  static bool firstTick = true;
  if(firstTick) {
    firstTick = false;
    Log::i("[WebApi] main loop first tick");
    }
  if(!s_isRunning) {
    emscripten_cancel_main_loop();
    return;
    }

  // An exception escaping this callback kills the requestAnimationFrame
  // chain and the engine dies silently (page keeps running, canvas stays
  // black). Log and keep the loop alive instead.
  static int iter = 0;
  ++iter;
  EM_ASM({ window.__ml = Array.of($0, 1); }, iter);
  try {
    if(s_appCallback != nullptr) {
      s_appCallback->onTimer();
      }
    EM_ASM({ window.__ml[1] = 2; });

    if(s_currentWindow != nullptr && s_currentWindow->owner != nullptr) {
      double cssW = 0, cssH = 0;
      if(emscripten_get_element_css_size("#canvas", &cssW, &cssH) >= 0) {
        uint32_t curW = static_cast<uint32_t>(cssW);
        uint32_t curH = static_cast<uint32_t>(cssH);

        // self-healing: if the canvas drawing buffer drifted away from
        // css*dpr (a resize dispatch was lost or partially applied), force
        // a new resize event even when the tracked window size matches
        bool canvasDrift = false;
        int cw = 0, ch = 0;
        if(emscripten_get_canvas_element_size("#canvas", &cw, &ch) == EMSCRIPTEN_RESULT_SUCCESS) {
          const double dpr = emscripten_get_device_pixel_ratio();
          const int expW = std::max(1, int(curW * dpr + 0.5));
          const int expH = std::max(1, int(curH * dpr + 0.5));
          canvasDrift = (std::abs(cw - expW) > 2) || (std::abs(ch - expH) > 2);
          }

        if((curW > 0 && curH > 0 && (curW != s_currentWindow->width || curH != s_currentWindow->height)) || canvasDrift) {
          if(canvasDrift)
            Log::i("[WebApi] canvas drift detected: buffer=", cw, "x", ch, " css=", curW, "x", curH);
          s_currentWindow->width  = curW;
          s_currentWindow->height = curH;
          SizeEvent e{int(curW), int(curH)};
          WebApi::dispatchResize(*s_currentWindow->owner, e);
          }
        }

      EM_ASM({ window.__ml[1] = 3; });
      WebApi::dispatchRender(*s_currentWindow->owner);
      EM_ASM({ window.__ml[1] = 4; });
      }
    }
  catch(const std::exception& e) {
    Log::e("[WebApi] main loop exception: ", e.what());
    }
  catch(...) {
    Log::e("[WebApi] main loop exception: unknown (non-std)");
    }
  }

int WebApi::implExec(AppCallBack& cb) {
  s_appCallback = &cb;
  s_isRunning   = true;
  // no simulated infinite loop: unwinding main() through wasm frames would
  // run destructors of the engine state; main() returns instead
  emscripten_set_main_loop(mainLoopIteration, 0, 0);
  // ?uncapped: frames are not tied to the display refresh (benchmarking)
  if(EM_ASM_INT({ return location.search.indexOf('uncapped')>=0 ? 1 : 0; }))
    emscripten_set_main_loop_timing(EM_TIMING_SETIMMEDIATE, 1);
  return 0;
  }

void WebApi::implProcessEvents(AppCallBack& cb) {
  cb.onTimer();
  }

void WebApi::onCanvasResize(int width, int height) {
  if(s_currentWindow != nullptr && s_currentWindow->owner != nullptr) {
    s_currentWindow->width  = uint32_t(std::max(1, width));
    s_currentWindow->height = uint32_t(std::max(1, height));
    SizeEvent e{width, height};
    dispatchResize(*s_currentWindow->owner, e);
    }
  }

void WebApi::onMouseEvent(int type, int button, int x, int y) {
  if(s_currentWindow == nullptr || s_currentWindow->owner == nullptr)
    return;

  Event::MouseButton btn = Event::ButtonNone;
  if(button == 0) btn = Event::ButtonLeft;
  else if(button == 1) btn = Event::ButtonMid;
  else if(button == 2) btn = Event::ButtonRight;

  MouseEvent e(x, y, btn, Event::M_NoModifier);
  if(type == 0) {
    dispatchMouseDown(*s_currentWindow->owner, e);
    }
  else if(type == 1) {
    dispatchMouseUp(*s_currentWindow->owner, e);
    }
  else if(type == 2) {
    dispatchMouseMove(*s_currentWindow->owner, e);
    }
  }

void WebApi::onKeyEvent(int type, int keyCode) {
  if(s_currentWindow == nullptr || s_currentWindow->owner == nullptr)
    return;

  KeyEvent e(static_cast<Event::KeyType>(keyCode), Event::M_NoModifier);
  if(type == 0) {
    dispatchKeyDown(*s_currentWindow->owner, e, uint32_t(keyCode));
    } else {
    dispatchKeyUp(*s_currentWindow->owner, e, uint32_t(keyCode));
    }
  }

void WebApi::onFrameTick() {
  if(s_currentWindow != nullptr && s_currentWindow->owner != nullptr) {
    dispatchRender(*s_currentWindow->owner);
    }
  }

#endif // __EMSCRIPTEN__
