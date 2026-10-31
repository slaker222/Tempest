#pragma once
#if defined(__ANDROID__)

#include <Tempest/SystemApi>
#include <atomic>
#include <mutex>
#include <vector>
#include <condition_variable>

struct ANativeWindow;

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

    struct TouchEvent {
      int   action;
      float x;
      float y;
      int   id;
      };

    struct KeyEventData {
      int action;
      int keyCode;
      };

    struct MouseEventData {
      int action;
      int button;
      int x;
      int y;
      };

    using TouchHook = bool (*)(int action, float x, float y, int id);
    static void setTouchHook(TouchHook hook);

    static void  setNativeWindow(ANativeWindow* window);
    static void* nativeWindow();
    static void  notifySurfaceChanged(int width, int height);
    static void  setDensity(float density); // screen density (px per dp), for the interface scale
    static void  pushTouchEvent(int action, float x, float y, int id);
    static void  pushKeyEvent(int action, int keyCode);
    static void  setDeviceLostHandler(void (*fn)()); // the gpu driver lost the device (a hang): the app lowers its settings and ends
    static void  pushWebKey(int action, const char* code); // 0 down, 1 up; code: KeyboardEvent.code of the web page's touch layer
    static void  pushMouseEvent(int action, int button, int x, int y);

  private:
    AndroidApi();
    ~AndroidApi() override;

    Window*  implCreateWindow(Tempest::Window *owner, uint32_t width, uint32_t height) override;
    Window*  implCreateWindow(Tempest::Window *owner, ShowMode sm) override;
    void     implDestroyWindow(Window* w) override;
    void     implExit() override;

    Rect     implWindowClientRect(SystemApi::Window *w) override;
    bool     implSetAsFullscreen(SystemApi::Window *w, bool fullScreen) override;
    bool     implIsFullscreen(SystemApi::Window *w) override;
    float    implUiScale(SystemApi::Window *w) override;

    void     implSetCursorPosition(SystemApi::Window *w, int x, int y) override;
    void     implShowCursor(SystemApi::Window *w, CursorShape cursor) override;

    bool     implIsRunning() override;
    int      implExec(AppCallBack& cb) override;
    void     implProcessEvents(AppCallBack& cb) override;

    void     implSetWindowTitle(SystemApi::Window *w, const char* utf8) override;

    static std::atomic_bool          isExit;
    static ANativeWindow*            currentNativeWindow;
    static Tempest::Window*          currentOwner;
    static std::mutex                windowSync;
    static std::condition_variable   windowCv;

    static std::mutex                  eventSync;
    static std::vector<TouchEvent>     touchEvents;
    static std::vector<KeyEventData>   keyEvents;
    static std::vector<MouseEventData> mouseEvents;

    static float                     touchStartX;
    static float                     touchStartY;
    static uint64_t                  touchStartTime;
    static bool                      touchActive;
    static TouchHook                 touchHook;

  friend class SystemApi;
  };

}

#endif
