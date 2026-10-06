import os
import librosa
import numpy as np
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
import moviepy.video.fx as vfx

# ================= НАСТРОЙКИ =================
BEATS_STEP = 2             # Каждые 2 удара меняем фото
VIDEO_W, VIDEO_H = 1080, 1920  # Вертикальный формат 9:16
ZOOM_SPEED = 0.1           # Насколько сильно приближается фото
# =============================================

def create_pure_vertical_video():
    # 1. Поиск файлов
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav'))]
    
    if not images or not audio_files:
        print("Ошибка: Нет фото или mp3 в папке!")
        return
    
    audio_path = audio_files[0]

    # 2. Анализ ритма
    print("Считаем биты в музыке...")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Определяем интервалы между ударами
    # (Берем среднюю длительность удара, если фоток больше, чем музыки)
    selected_beats = beat_times[::BEATS_STEP]
    durations = np.diff(selected_beats)
    avg_duration = np.mean(durations) if len(durations) > 0 else 1.0

    # 3. Подготовка фото
    print(f"Монтируем {len(images)} кадров...")
    clips = []
    
    for i in range(len(images)):
        # Если удары в музыке закончились, используем средний темп
        dur = durations[i] if i < len(durations) else avg_duration
        img_path = images[i]

        clip = ImageClip(img_path).with_duration(dur)

        # ЛОГИКА "ЗАПОЛНИТЬ ВЕСЬ ЭКРАН"
        target_ratio = VIDEO_W / VIDEO_H
        img_ratio = clip.w / clip.h

        if img_ratio > target_ratio:
            clip = clip.resized(height=VIDEO_H)
        else:
            clip = clip.resized(width=VIDEO_W)

        clip = clip.cropped(x_center=clip.w/2, y_center=clip.h/2, width=VIDEO_W, height=VIDEO_H)

        # Эффект плавного наезда (Zoom)
        def zoom_in(t):
            return 1 + ZOOM_SPEED * (t / dur)

        clip = clip.resized(zoom_in).with_position("center")
        clips.append(clip)

    # 4. Сборка видео
    print("Склеиваем клипы...")
    final_video = concatenate_videoclips(clips, method="compose")
    
    # 5. Работа с аудио (Зацикливание)
    print("Настраиваем аудио...")
    audio_clip = AudioFileClip(audio_path)
    
    if final_video.duration > audio_clip.duration:
        print("Музыка короче видео. Зацикливаю аудио...")
        # Зацикливаем аудио до длины видео
        audio_clip = audio_clip.with_effects([vfx.loop(duration=final_video.duration)])
    else:
        audio_clip = audio_clip.with_duration(final_video.duration)
    
    final_video = final_video.with_audio(audio_clip)

    # Сохраняем
    output_name = "STYLISH_REELS.mp4"
    print(f"Рендеринг в {output_name}...")
    
    try:
        final_video.write_videofile(
            output_name, 
            fps=30, 
            codec="libx264", 
            audio_codec="aac",
            threads=4,
            logger='bar'
        )
    finally:
        # Явное закрытие всех ресурсов для предотвращения WinError 6
        final_video.close()
        audio_clip.close()
        for c in clips:
            c.close()
    
    print("\n--- ГОТОВО! ---")
    print(f"Ищи файл: {output_name}")

if __name__ == "__main__":
    create_pure_vertical_video()