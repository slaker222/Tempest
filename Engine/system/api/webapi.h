#pragma once

#if defined(__EMSCRIPTEN__)

#include <Tempest/SystemApi>
#include <cstdint>

namespace Tempest {

class WebApi final : public SystemApi {
  public:
    using SystemApi::dispatchRender;
    using SystemApi::dispatchResize;
    using SystemApi::dispatchClose;
    using SystemApi::dispatchMouseDown;
    using SystemApi::dispatchMouseUp;
    using SystemApi::dispatchMouseMove;
    using SystemApi::dispatchMouseWheel;
    using SystemApi::dispatchKeyDown;
    using SystemApi::dispatchKeyUp;
    using AppCallBack = SystemApi::AppCallBack;

    static void onCanvasResize(int width, int height);
    static void onMouseEvent(int type, int button, int x, int y);
    static void onKeyEvent(int type, int keyCode);
    static void onFrameTick();

  private:
    struct Window;

    WebApi();
    ~WebApi() override;

    SystemApi::Window* implCreateWindow(Tempest::Window* owner, uint32_t width, uint32_t height) override;
    SystemApi::Window* implCreateWindow(Tempest::Window* owner, ShowMode sm) override;
    void               implDestroyWindow(SystemApi::Window* w) override;
    void               implExit() override;

    Rect               implWindowClientRect(SystemApi::Window* w) override;

    bool               implSetAsFullscreen(SystemApi::Window* w, bool fullScreen) override;
    bool               implIsFullscreen(SystemApi::Window* w) override;
    float              implUiScale(SystemApi::Window* w) override;

    void               implSetWindowTitle(SystemApi::Window* w, const char* utf8) override;
    void               implSetCursorPosition(SystemApi::Window* w, int x, int y) override;
    void               implShowCursor(SystemApi::Window* w, CursorShape show) override;

    bool               implIsRunning() override;
    int                implExec(AppCallBack& cb) override;
    void               implProcessEvents(AppCallBack& cb) override;

    friend class SystemApi;
  };

}

#endif
