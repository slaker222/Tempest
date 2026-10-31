#if defined(__ANDROID__)

#include "androidapi.h"

#include <Tempest/Window>
#include <Tempest/Application>
#include <Tempest/Log>
#include <android/native_window.h>
#include <cmath>
#include <cstring>
#include <thread>
#include <chrono>

using namespace Tempest;

std::atomic_bool          AndroidApi::isExit{false};
ANativeWindow*            AndroidApi::currentNativeWindow = nullptr;
Tempest::Window*          AndroidApi::currentOwner        = nullptr;
std::mutex                AndroidApi::windowSync;
std::condition_variable   AndroidApi::windowCv;

std::mutex                AndroidApi::eventSync;
std::vector<AndroidApi::TouchEvent>     AndroidApi::touchEvents;
std::vector<AndroidApi::KeyEventData>   AndroidApi::keyEvents;
std::vector<AndroidApi::MouseEventData> AndroidApi::mouseEvents;

float                     AndroidApi::touchStartX     = 0.0f;
float                     AndroidApi::touchStartY     = 0.0f;
uint64_t                  AndroidApi::touchStartTime  = 0;
bool                      AndroidApi::touchActive     = false;
AndroidApi::TouchHook     AndroidApi::touchHook       = nullptr;

void AndroidApi::setTouchHook(TouchHook hook) {
  touchHook = hook;
  }

AndroidApi::AndroidApi() {
  }

AndroidApi::~AndroidApi() {
  setNativeWindow(nullptr);
  }

void AndroidApi::setNativeWindow(ANativeWindow* window) {
  std::lock_guard<std::mutex> lk(windowSync);
  if(currentNativeWindow != nullptr && currentNativeWindow != window) {
    ANativeWindow_release(currentNativeWindow);
    }
  currentNativeWindow = window;
  if(currentNativeWindow != nullptr) {
    ANativeWindow_acquire(currentNativeWindow);
    int32_t w = ANativeWindow_getWidth(currentNativeWindow);
    int32_t h = ANativeWindow_getHeight(currentNativeWindow);
    Log::i("AndroidApi::setNativeWindow: ", window, " size: ", w, "x", h);
    } else {
    Log::i("AndroidApi::setNativeWindow: null");
    }
  windowCv.notify_all();

  if(currentOwner != nullptr && currentNativeWindow != nullptr) {
    int32_t w = ANativeWindow_getWidth(currentNativeWindow);
    int32_t h = ANativeWindow_getHeight(currentNativeWindow);
    if(w > 0 && h > 0) {
      SizeEvent ev(w, h);
      SystemApi::dispatchResize(*currentOwner, ev);
      }
    }
  }

void* AndroidApi::nativeWindow() {
  return currentNativeWindow;
  }

void AndroidApi::notifySurfaceChanged(int width, int height) {
  Log::i("AndroidApi::notifySurfaceChanged: ", width, "x", height);
  if(currentOwner != nullptr && width > 0 && height > 0) {
    SizeEvent ev(width, height);
    SystemApi::dispatchResize(*currentOwner, ev);
    }
  }

void AndroidApi::pushTouchEvent(int action, float x, float y, int id) {
  std::lock_guard<std::mutex> lk(eventSync);
  touchEvents.push_back({action, x, y, id});
  }

namespace {
using namespace Tempest;
static Event::KeyType mapWebKeyCode(const char* code) {
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
  if(std::strcmp(code, "CapsLock") == 0) return Event::K_CapsLock;
  if(std::strcmp(code, "ShiftLeft") == 0) return Event::K_LShift;
  if(std::strcmp(code, "ShiftRight") == 0) return Event::K_RShift;
  if(std::strcmp(code, "ControlLeft") == 0) return Event::K_LControl;
  if(std::strcmp(code, "ControlRight") == 0) return Event::K_RControl;
  if(std::strcmp(code, "AltLeft") == 0) return Event::K_LAlt;
  if(std::strcmp(code, "AltRight") == 0) return Event::K_RAlt;
  if(std::strcmp(code, "Delete") == 0) return Event::K_Delete;
  if(std::strcmp(code, "Insert") == 0) return Event::K_Insert;
  if(std::strcmp(code, "Home") == 0) return Event::K_Home;
  if(std::strcmp(code, "End") == 0) return Event::K_End;
  if(std::strncmp(code, "Digit", 5) == 0 && code[5]>='0' && code[5]<='9' && code[6]=='\0')
    return Event::KeyType(Event::K_0 + (code[5]-'0'));
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
}

void AndroidApi::pushWebKey(int action, const char* code) {
  const auto k = mapWebKeyCode(code);
  if(k==Event::K_NoKey)
    return;
  pushKeyEvent(action, 1000+int(k));
  }

void AndroidApi::pushKeyEvent(int action, int keyCode) {
  std::lock_guard<std::mutex> lk(eventSync);
  keyEvents.push_back({action, keyCode});
  }

void AndroidApi::pushMouseEvent(int action, int button, int x, int y) {
  std::lock_guard<std::mutex> lk(eventSync);
  mouseEvents.push_back({action, button, x, y});
  }

SystemApi::Window* AndroidApi::implCreateWindow(Tempest::Window* owner, uint32_t /*width*/, uint32_t /*height*/) {
  Log::i("AndroidApi::implCreateWindow waiting for ANativeWindow...");
  std::unique_lock<std::mutex> lk(windowSync);
  windowCv.wait(lk, []{ return currentNativeWindow != nullptr || isExit.load(); });

  if(isExit.load() || currentNativeWindow == nullptr) {
    Log::e("AndroidApi::implCreateWindow: cancelled or null window");
    return nullptr;
    }

  currentOwner = owner;
  int32_t w = ANativeWindow_getWidth(currentNativeWindow);
  int32_t h = ANativeWindow_getHeight(currentNativeWindow);
  Log::i("AndroidApi::implCreateWindow ready with ANativeWindow: ", currentNativeWindow, " (", w, "x", h, ")");
  return reinterpret_cast<SystemApi::Window*>(currentNativeWindow);
  }

SystemApi::Window* AndroidApi::implCreateWindow(Tempest::Window* owner, ShowMode /*sm*/) {
  return implCreateWindow(owner, 1920, 1080);
  }

void AndroidApi::implDestroyWindow(Window* /*w*/) {
  std::lock_guard<std::mutex> lk(windowSync);
  currentOwner = nullptr;
  }

void AndroidApi::implExit() {
  isExit.store(true);
  windowCv.notify_all();
  }

Rect AndroidApi::implWindowClientRect(SystemApi::Window* w) {
  ANativeWindow* win = reinterpret_cast<ANativeWindow*>(w ? w : reinterpret_cast<SystemApi::Window*>(currentNativeWindow));
  if(win != nullptr) {
    int32_t width  = ANativeWindow_getWidth(win);
    int32_t height = ANativeWindow_getHeight(win);
    if(width > 0 && height > 0)
      return Rect(0, 0, width, height);
    }
  return Rect(0, 0, 1920, 1080);
  }

bool AndroidApi::implSetAsFullscreen(SystemApi::Window* /*w*/, bool /*fullScreen*/) {
  return true;
  }

bool AndroidApi::implIsFullscreen(SystemApi::Window* /*w*/) {
  return true;
  }

void AndroidApi::implSetCursorPosition(SystemApi::Window* /*w*/, int /*x*/, int /*y*/) {
  }

void AndroidApi::implShowCursor(SystemApi::Window* /*w*/, CursorShape /*cursor*/) {
  }

bool AndroidApi::implIsRunning() {
  return !isExit.load();
  }

int AndroidApi::implExec(AppCallBack& cb) {
  Log::i("AndroidApi::implExec game render loop started");
  while(!isExit.load()) {
    implProcessEvents(cb);

    if(cb.onTimer() == 0) {
      std::this_thread::yield();
      }

    if(currentOwner != nullptr && currentNativeWindow != nullptr) {
      try {
        SystemApi::dispatchRender(*currentOwner);
        }
      catch(const std::exception& e) {
        Log::e("dispatchRender error: ", e.what());
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
      catch(...) {
        Log::e("dispatchRender unknown error");
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
      } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
      }
    }
  Log::i("AndroidApi::implExec game render loop finished");
  return 0;
  }

void AndroidApi::implProcessEvents(AppCallBack& /*cb*/) {
  std::vector<TouchEvent>     tEvents;
  std::vector<KeyEventData>   kEvents;
  std::vector<MouseEventData> mEvents;
  {
    std::lock_guard<std::mutex> lk(eventSync);
    tEvents.swap(touchEvents);
    kEvents.swap(keyEvents);
    mEvents.swap(mouseEvents);
    }

  if(currentOwner == nullptr)
    return;

  // Mouse events handling
  for(const auto& me : mEvents) {
    Event::MouseButton mb = (me.button == 1) ? Event::ButtonLeft : (me.button == 2 ? Event::ButtonRight : Event::ButtonNone);
    if(me.action == 0) {
      MouseEvent md(me.x, me.y, mb, Event::M_NoModifier, 0, 0, Event::MouseDown);
      SystemApi::dispatchMouseDown(*currentOwner, md);
      } else {
      MouseEvent mu(me.x, me.y, mb, Event::M_NoModifier, 0, 0, Event::MouseUp);
      SystemApi::dispatchMouseUp(*currentOwner, mu);
      }
    }

  // Key events handling
  for(const auto& ke : kEvents) {
    Event::KeyType k = Event::K_NoKey;
    if(ke.keyCode>=1000)
      k = Event::KeyType(ke.keyCode-1000); // a key of the web page's touch layer
    else
    switch(ke.keyCode) {
      case 4:   k = Event::K_ESCAPE; break; // KEYCODE_BACK
      case 111: k = Event::K_ESCAPE; break; // KEYCODE_ESCAPE
      case 19:  k = Event::K_Up;     break; // KEYCODE_DPAD_UP
      case 20:  k = Event::K_Down;   break; // KEYCODE_DPAD_DOWN
      case 21:  k = Event::K_Left;   break; // KEYCODE_DPAD_LEFT
      case 22:  k = Event::K_Right;  break; // KEYCODE_DPAD_RIGHT
      case 23:
      case 66:  k = Event::K_Return; break; // KEYCODE_ENTER / DPAD_CENTER
      case 62:  k = Event::K_Space;  break; // KEYCODE_SPACE
      default:  break;
      }
    if(k != Event::K_NoKey) {
      KeyEvent ev(k, 0, Event::M_NoModifier, ke.action == 0 ? Event::KeyDown : Event::KeyUp);
      if(ke.action == 0)
        SystemApi::dispatchKeyDown(*currentOwner, ev, ke.keyCode);
      else
        SystemApi::dispatchKeyUp(*currentOwner, ev, ke.keyCode);
      }
    }

  // Touch events handling
  for(const auto& te : tEvents) {
    if(touchHook != nullptr && touchHook(te.action, te.x, te.y, te.id)) {
      continue;
      }

    int x = static_cast<int>(te.x);
    int y = static_cast<int>(te.y);

    if(te.action == 0 || te.action == 5) { // ACTION_DOWN / ACTION_POINTER_DOWN
      touchStartX    = te.x;
      touchStartY    = te.y;
      touchStartTime = Application::tickCount();
      touchActive    = true;

      MouseEvent mv(x, y, Event::ButtonNone, Event::M_NoModifier, 0, te.id, Event::MouseMove);
      SystemApi::dispatchMouseMove(*currentOwner, mv);

      MouseEvent md(x, y, Event::ButtonLeft, Event::M_NoModifier, 0, te.id, Event::MouseDown);
      SystemApi::dispatchMouseDown(*currentOwner, md);
      }
    else if(te.action == 2) { // ACTION_MOVE
      MouseEvent mv(x, y, Event::ButtonLeft, Event::M_NoModifier, 0, te.id, Event::MouseMove);
      SystemApi::dispatchMouseMove(*currentOwner, mv);
      }
    else if(te.action == 1 || te.action == 6 || te.action == 3) { // ACTION_UP / ACTION_POINTER_UP / ACTION_CANCEL
      MouseEvent mu(x, y, Event::ButtonLeft, Event::M_NoModifier, 0, te.id, Event::MouseUp);
      SystemApi::dispatchMouseUp(*currentOwner, mu);
      touchActive = false;
      }
    }
  }

void AndroidApi::implSetWindowTitle(SystemApi::Window* /*w*/, const char* /*utf8*/) {
  }

#endif
