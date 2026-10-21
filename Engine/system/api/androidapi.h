#pragma once

#include <Tempest/SystemApi>
#include <atomic>

namespace Tempest {

class AndroidApi final: public SystemApi {
  public:
    using SystemApi::dispatchRender;
    using SystemApi::dispatchResize;
    using SystemApi::dispatchClose;
    using SystemApi::dispatchMouseDown;
    using SystemApi::dispatchMouseUp;
    using SystemApi::dispatchMouseMove;
    using SystemApi::dispatchKeyDown;
    using SystemApi::dispatchKeyUp;

    // Временные заглушки для Этапа 1 (будут полноценно реализованы на Этапе 2)
    static void  setNativeWindow(void* window);
    static void* nativeWindow();

  private:
    AndroidApi();

    Window*  implCreateWindow(Tempest::Window *owner, uint32_t width, uint32_t height) override;
    Window*  implCreateWindow(Tempest::Window *owner, ShowMode sm) override;
    void     implDestroyWindow(Window* w) override;
    void     implExit() override;

    Rect     implWindowClientRect(SystemApi::Window *w) override;
    bool     implSetAsFullscreen(SystemApi::Window *w, bool fullScreen) override;
    bool     implIsFullscreen(SystemApi::Window *w) override;

    void     implSetCursorPosition(SystemApi::Window *w, int x, int y) override;
    void     implShowCursor(SystemApi::Window *w, CursorShape cursor) override;

    bool     implIsRunning() override;
    int      implExec(AppCallBack& cb) override;
    void     implProcessEvents(AppCallBack& cb) override;

    void     implSetWindowTitle(SystemApi::Window *w, const char* utf8) override;

    static std::atomic_bool isExit;
    static void*            currentNativeWindow;

  friend class SystemApi;
  };

}
