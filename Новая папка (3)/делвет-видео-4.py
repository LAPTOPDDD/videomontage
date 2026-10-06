import os
import librosa
import numpy as np
from moviepy import ImageClip, VideoClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips

# ================= НАСТРОЙКИ (VERTICAL) =================
BEATS_STEP = 2            # Каждые сколько ударов менять фото
VIDEO_W, VIDEO_H = 1080, 1920  # Разрешение для смартфона (9:16)
# =========================================================

def create_mobile_video():
    # 1. Поиск файлов
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav'))]
    
    if not images or not audio_files:
        print("Ошибка: Добавьте фото и mp3 в папку!")
        return
    
    audio_path = audio_files[0]

    # 2. Анализ аудио
    print("Анализируем ритм для смартфона...")
    y, sr = librosa.load(audio_path)
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    selected_beats = beat_times[::BEATS_STEP]
    durations = np.diff(selected_beats)

    # Анализ частот для фона
    S = np.abs(librosa.stft(y))
    bass_ch = np.mean(S[:20, :], axis=0)
    mids_ch = np.mean(S[20:100, :], axis=0)
    highs_ch = np.mean(S[100:, :], axis=0)

    def norm(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return np.power(arr, 1.4) * 255 # Чуть мягче пульсация

    bass = norm(bass_ch)
    mids = norm(mids_ch)
    highs = norm(highs_ch)
    color_times = np.linspace(0, librosa.get_duration(y=y, sr=sr), len(bass))

    # 3. Функция фона (Вертикальная цветомузыка)
    def make_bg_frame(t):
        idx = np.searchsorted(color_times, t)
        if idx >= len(bass): idx = len(bass)-1
        # RGB пульсация
        color = [int(bass[idx]), int(mids[idx]*0.4), int(highs[idx])]
        return np.full((VIDEO_H, VIDEO_W, 3), color, dtype='uint8')

    # 4. Сборка видеоряда
    print(f"Подготовка {len(images)} вертикальных слайдов...")
    clips = []
    num_to_make = min(len(images), len(durations))

    for i in range(num_to_make):
        dur = durations[i]
        img_path = images[i]

        # Плавный Zoom (наезд камеры)
        def zoom(t):
            return 1 + 0.1 * (t / dur)

        # Подгоняем фото под ширину смартфона
        img_clip = ImageClip(img_path).with_duration(dur)
        
        # Умный ресайз: делаем фото по ширине экрана
        img_clip = img_clip.resized(width=VIDEO_W) 
        
        # Если фото после ресайза всё еще слишком высокое, подрезаем под экран
        if img_clip.h > VIDEO_H:
            img_clip = img_clip.cropped(y_center=img_clip.h/2, height=VIDEO_H)
            
        img_clip = img_clip.resized(zoom).with_position("center")
        clips.append(img_clip)

    # 5. Финальный монтаж
    slideshow = concatenate_videoclips(clips, method="compose")
    
    # Фон цветомузыки
    bg_layer = VideoClip(make_bg_frame, duration=slideshow.duration)
    
    # Накладываем фото на фон. 
    # Теперь фото будет крупным в центре, а фон будет виден сверху и снизу
    final_video = CompositeVideoClip([bg_layer, slideshow.with_position("center")], size=(VIDEO_W, VIDEO_H))
    
    audio_clip = AudioFileClip(audio_path).with_duration(final_video.duration)
    final_video = final_video.with_audio(audio_clip)

    # 6. Сохранение
    output_name = "MOBILE_REELS_VERSION.mp4"
    print("Рендеринг вертикального видео...")
    final_video.write_videofile(output_name, fps=24, codec="libx264", audio_codec="aac")
    print(f"\nГОТОВО! Видео для смартфона: {output_name}")

if __name__ == "__main__":
    create_mobile_video()