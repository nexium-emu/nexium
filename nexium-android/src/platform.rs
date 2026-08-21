use android_activity::AndroidApp;
use jni::objects::JObject;
use jni::{jni_sig, jni_str, JavaVM};

fn with_env<T>(f: impl FnOnce(&mut jni::Env) -> jni::errors::Result<T>) -> Option<T> {
    let vm = JavaVM::singleton().ok()?;
    vm.attach_current_thread(|env: &mut jni::Env| f(env)).ok()
}

pub fn has_all_files_access() -> bool {
    with_env(|env| {
        let class = env.find_class(jni_str!("android/os/Environment"))?;
        env.call_static_method(
            &class,
            jni_str!("isExternalStorageManager"),
            jni_sig!(() -> boolean),
            &[],
        )?
        .z()
    })
    .unwrap_or(false)
}

fn hide_system_bars_now(activity_ptr: *mut std::ffi::c_void) {
    if activity_ptr.is_null() {
        return;
    }
    let done = with_env(|env| {
        let activity = unsafe { JObject::from_raw(env, activity_ptr as jni::sys::jobject) };
        let window = env
            .call_method(
                &activity,
                jni_str!("getWindow"),
                jni_sig!(() -> android.view.Window),
                &[],
            )?
            .l()?;
        std::mem::forget(activity);

        let params = env
            .call_method(
                &window,
                jni_str!("getAttributes"),
                jni_sig!(() -> android.view.WindowManager::LayoutParams),
                &[],
            )?
            .l()?;
        env.set_field(
            &params,
            jni_str!("layoutInDisplayCutoutMode"),
            jni_sig!(int),
            3i32.into(),
        )?;
        env.call_method(
            &window,
            jni_str!("setAttributes"),
            jni_sig!((android.view.WindowManager::LayoutParams) -> void),
            &[(&params).into()],
        )?;

        let controller = env
            .call_method(
                &window,
                jni_str!("getInsetsController"),
                jni_sig!(() -> android.view.WindowInsetsController),
                &[],
            )?
            .l()?;
        if controller.is_null() {
            return Ok(false);
        }

        let types_class = env.find_class(jni_str!("android/view/WindowInsets$Type"))?;
        let bars = env
            .call_static_method(
                &types_class,
                jni_str!("systemBars"),
                jni_sig!(() -> int),
                &[],
            )?
            .i()?;

        env.call_method(
            &controller,
            jni_str!("hide"),
            jni_sig!((int) -> void),
            &[bars.into()],
        )?;
        env.call_method(
            &controller,
            jni_str!("setSystemBarsBehavior"),
            jni_sig!((int) -> void),
            &[2i32.into()],
        )?;
        Ok(true)
    })
    .unwrap_or(false);
    if !done {
        log::warn!("immersive: could not hide system bars");
    }
}

pub fn hide_system_bars(app: &AndroidApp) {
    let ptr = app.activity_as_ptr() as usize;
    app.run_on_java_main_thread(Box::new(move || {
        hide_system_bars_now(ptr as *mut std::ffi::c_void);
    }));
}

pub fn request_all_files_access(app: &AndroidApp) -> bool {
    let activity_ptr = app.activity_as_ptr();
    if activity_ptr.is_null() {
        return false;
    }
    with_env(|env| {
        let activity = unsafe { JObject::from_raw(env, activity_ptr as jni::sys::jobject) };
        let action = env.new_string(
            "android.settings.MANAGE_APP_ALL_FILES_ACCESS_PERMISSION",
        )?;
        let package = env.new_string("package:dev.nexium.emu")?;

        let uri_class = env.find_class(jni_str!("android/net/Uri"))?;
        let uri = env
            .call_static_method(
                &uri_class,
                jni_str!("parse"),
                jni_sig!((java.lang.String) -> android.net.Uri),
                &[(&package).into()],
            )?
            .l()?;

        let intent_class = env.find_class(jni_str!("android/content/Intent"))?;
        let intent = env.new_object(
            &intent_class,
            jni_sig!((java.lang.String, android.net.Uri) -> void),
            &[(&action).into(), (&uri).into()],
        )?;

        env.call_method(
            &activity,
            jni_str!("startActivity"),
            jni_sig!((android.content.Intent) -> void),
            &[(&intent).into()],
        )?;
        std::mem::forget(activity);
        Ok(true)
    })
    .unwrap_or(false)
}
