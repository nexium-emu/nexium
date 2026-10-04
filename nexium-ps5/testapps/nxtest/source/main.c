#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <switch.h>

#define FB_WIDTH 1280
#define FB_HEIGHT 720
#define SAMPLERATE 48000
#define AUDIO_FRAMES 1600
#define AUDIO_BUFFERS 3
#define COUNTER_PATH "sdmc:/nxtest-launches.txt"

static u32 rgba(u8 r, u8 g, u8 b) {
    return RGBA8_MAXALPHA(r, g, b);
}

static void fill_rect(u32 *fb, u32 stride, int x0, int y0, int w, int h, u32 color) {
    for (int y = y0; y < y0 + h; y++) {
        if (y < 0 || y >= FB_HEIGHT) continue;
        u32 *row = fb + y * stride;
        for (int x = x0; x < x0 + w; x++) {
            if (x >= 0 && x < FB_WIDTH) row[x] = color;
        }
    }
}

static void debug_line(const char *text) {
    svcOutputDebugString(text, strlen(text));
}

static u32 bump_launch_counter(void) {
    u32 count = 0;
    FILE *f = fopen(COUNTER_PATH, "r");
    if (f) {
        if (fscanf(f, "%u", &count) != 1) count = 0;
        fclose(f);
    }
    count++;
    f = fopen(COUNTER_PATH, "w");
    if (f) {
        fprintf(f, "%u\n", count);
        fclose(f);
    }
    return count;
}

static void fill_tone(s16 *dest, u32 frames, float freq, float *phase) {
    for (u32 i = 0; i < frames; i++) {
        s16 sample = 0;
        if (freq > 0.0f) {
            sample = (s16)(0.25f * 32767.0f * sinf(*phase));
            *phase += 2.0f * (float)M_PI * freq / SAMPLERATE;
            if (*phase > 2.0f * (float)M_PI) *phase -= 2.0f * (float)M_PI;
        }
        dest[i * 2] = sample;
        dest[i * 2 + 1] = sample;
    }
}

static const u64 kButtons[16] = {
    HidNpadButton_A, HidNpadButton_B, HidNpadButton_X, HidNpadButton_Y,
    HidNpadButton_L, HidNpadButton_R, HidNpadButton_ZL, HidNpadButton_ZR,
    HidNpadButton_Up, HidNpadButton_Down, HidNpadButton_Left, HidNpadButton_Right,
    HidNpadButton_StickL, HidNpadButton_StickR, HidNpadButton_Minus, HidNpadButton_Plus,
};

int main(int argc, char *argv[]) {
    char line[160];
    u32 launches = bump_launch_counter();
    snprintf(line, sizeof(line), "nxtest: start, launch count %u", launches);
    debug_line(line);

    NWindow *win = nwindowGetDefault();
    Framebuffer fb;
    framebufferCreate(&fb, win, FB_WIDTH, FB_HEIGHT, PIXEL_FORMAT_RGBA_8888, 2);
    framebufferMakeLinear(&fb);

    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);

    u32 data_size = AUDIO_FRAMES * 2 * sizeof(s16);
    u32 buffer_size = (data_size + 0xfff) & ~0xfff;
    AudioOutBuffer buffers[AUDIO_BUFFERS];
    s16 *audio_data[AUDIO_BUFFERS];
    bool audio_ok = R_SUCCEEDED(audoutInitialize()) && R_SUCCEEDED(audoutStartAudioOut());
    float phase = 0.0f;
    u64 audio_appended = 0;
    if (audio_ok) {
        for (int i = 0; i < AUDIO_BUFFERS; i++) {
            audio_data[i] = memalign(0x1000, buffer_size);
            memset(audio_data[i], 0, buffer_size);
            buffers[i].next = NULL;
            buffers[i].buffer = audio_data[i];
            buffers[i].buffer_size = buffer_size;
            buffers[i].data_size = data_size;
            buffers[i].data_offset = 0;
            fill_tone(audio_data[i], AUDIO_FRAMES, i == 0 ? 523.25f : 659.25f, &phase);
            audoutAppendAudioOutBuffer(&buffers[i]);
            audio_appended++;
        }
    }
    snprintf(line, sizeof(line), "nxtest: audio %s", audio_ok ? "started" : "unavailable");
    debug_line(line);

    u32 frame = 0;
    u64 last_buttons = 0;
    while (appletMainLoop()) {
        padUpdate(&pad);
        u64 held = padGetButtons(&pad);
        u64 down = padGetButtonsDown(&pad);
        HidAnalogStickState ls = padGetStickPos(&pad, 0);
        HidAnalogStickState rs = padGetStickPos(&pad, 1);
        if (down & HidNpadButton_Plus) break;
        if (held != last_buttons) {
            snprintf(line, sizeof(line), "nxtest: buttons %#llx lstick %d,%d rstick %d,%d", (unsigned long long)held,
                     ls.x, ls.y, rs.x, rs.y);
            debug_line(line);
            last_buttons = held;
        }

        if (audio_ok) {
            AudioOutBuffer *released = NULL;
            u32 count = 0;
            while (R_SUCCEEDED(audoutGetReleasedAudioOutBuffer(&released, &count)) && count > 0 && released) {
                float freq = (held & HidNpadButton_A) ? 440.0f : 0.0f;
                fill_tone((s16 *)released->buffer, AUDIO_FRAMES, freq, &phase);
                audoutAppendAudioOutBuffer(released);
                audio_appended++;
                released = NULL;
            }
        }

        u32 stride;
        u32 *buf = (u32 *)framebufferBegin(&fb, &stride);
        stride /= sizeof(u32);
        for (u32 y = 0; y < FB_HEIGHT; y++) {
            u32 *row = buf + y * stride;
            u8 g = (u8)((y * 255) / FB_HEIGHT);
            for (u32 x = 0; x < FB_WIDTH; x++) {
                row[x] = rgba((u8)((x + frame * 4) & 0xff), g, 96);
            }
        }
        for (int i = 0; i < 16; i++) {
            u32 color = (held & kButtons[i]) ? rgba(255, 230, 40) : rgba(30, 30, 30);
            fill_rect(buf, stride, 40 + i * 75, 600, 65, 80, color);
        }
        int lx = FB_WIDTH / 4 + (ls.x * 200) / JOYSTICK_MAX;
        int ly = FB_HEIGHT / 2 - (ls.y * 200) / JOYSTICK_MAX;
        int rx = 3 * FB_WIDTH / 4 + (rs.x * 200) / JOYSTICK_MAX;
        int ry = FB_HEIGHT / 2 - (rs.y * 200) / JOYSTICK_MAX;
        fill_rect(buf, stride, FB_WIDTH / 4 - 210, FB_HEIGHT / 2 - 210, 420, 420, rgba(20, 20, 60));
        fill_rect(buf, stride, 3 * FB_WIDTH / 4 - 210, FB_HEIGHT / 2 - 210, 420, 420, rgba(60, 20, 20));
        fill_rect(buf, stride, lx - 15, ly - 15, 30, 30, rgba(255, 255, 255));
        fill_rect(buf, stride, rx - 15, ry - 15, 30, 30, rgba(255, 255, 255));
        fill_rect(buf, stride, 0, 0, (int)((frame % 600) * FB_WIDTH / 600), 12, rgba(255, 255, 255));
        for (u32 i = 0; i < launches && i < 40; i++) {
            fill_rect(buf, stride, 40 + i * 30, 30, 22, 22, rgba(40, 220, 90));
        }
        if (audio_ok && (held & HidNpadButton_A)) fill_rect(buf, stride, FB_WIDTH - 90, 30, 50, 50, rgba(60, 120, 255));
        framebufferEnd(&fb);

        if (frame % 300 == 0) {
            snprintf(line, sizeof(line), "nxtest: frame %u audio buffers %llu", frame, (unsigned long long)audio_appended);
            debug_line(line);
        }
        frame++;
    }

    debug_line("nxtest: exiting");
    if (audio_ok) {
        audoutStopAudioOut();
        audoutExit();
    }
    framebufferClose(&fb);
    return 0;
}
