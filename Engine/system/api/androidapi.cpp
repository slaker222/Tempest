#include "androidapi.h"

#include <Tempest/Window>
#include <Tempest/Log>

using namespace Tempest;

std::atomic_bool AndroidApi::isExit{false};
void*            AndroidApi::currentNativeWindow = nullptr;

AndroidApi::AndroidApi() {
  }

void AndroidApi::setNativeWindow(void* window) {
  currentNativeWindow = window;
  }

void* AndroidApi::nativeWindow() {
  return currentNativeWindow;
  }

SystemApi::Window* AndroidApi::implCreateWindow(Tempest::Window* /*owner*/, uint32_t /*width*/, uint32_t /*height*/) {
  // STUB (Этап 1): временная заглушка дескриптора окна до Этапа 2
  return reinterpret_cast<SystemApi::Window*>(currentNativeWindow ? currentNativeWindow : reinterpret_cast<void*>(0x1));
  }

SystemApi::Window* AndroidApi::implCreateWindow(Tempest::Window* owner, ShowMode /*sm*/) {
  return implCreateWindow(owner, 1920, 1080);
  }

void AndroidApi::implDestroyWindow(Window* /*w*/) {
  // STUB (Этап 1)
  }

void AndroidApi::implExit() {
  isExit.store(true);
  }

Rect AndroidApi::implWindowClientRect(SystemApi::Window* /*w*/) {
  // STUB (Этап 1): возвращает базовое разрешение мобильного экрана
  return Rect(0, 0, 1920, 1080);
  }

bool AndroidApi::implSetAsFullscreen(SystemApi::Window* /*w*/, bool /*fullScreen*/) {
  return true;
  }

bool AndroidApi::implIsFullscreen(SystemApi::Window* /*w*/) {
  return true;
  }

void AndroidApi::implSetCursorPosition(SystemApi::Window* /*w*/, int /*x*/, int /*y*/) {
  // На сенсорном экране отсутствует аппаратный курсор мыши
  }

void AndroidApi::implShowCursor(SystemApi::Window* /*w*/, CursorShape /*cursor*/) {
  }

bool AndroidApi::implIsRunning() {
  return !isExit.load();
  }

int AndroidApi::implExec(AppCallBack& cb) {
  // STUB (Этап 1): в shared library не блокируем основной поток Android
  while(!isExit.load()) {
    implProcessEvents(cb);
    break;
    }
  return 0;
  }

void AndroidApi::implProcessEvents(AppCallBack& cb) {
  // STUB (Этап 1): опрос таймеров
  cb.onTimer();
  }

void AndroidApi::implSetWindowTitle(SystemApi::Window* /*w*/, const char* /*utf8*/) {
  }
