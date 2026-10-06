import os
import librosa
import numpy as np
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

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

    # 2. Анализ ритма (Только биты, без частот)
    print("Считаем биты в музыке...")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    selected_beats = beat_times[::BEATS_STEP]
    durations = np.diff(selected_beats)

    # 3. Подготовка фото (Кроп под экран смартфона)
    print(f"Монтируем {len(images)} кадров...")
    clips = []
    num_to_make = min(len(images), len(durations))

    for i in range(num_to_make):
        dur = durations[i]
        img_path = images[i]

        # Загружаем фото и задаем длительность
        clip = ImageClip(img_path).with_duration(dur)

        # ЛОГИКА "ЗАПОЛНИТЬ ВЕСЬ ЭКРАН"
        # Вычисляем пропорции
        target_ratio = VIDEO_W / VIDEO_H
        img_ratio = clip.w / clip.h

        if img_ratio > target_ratio:
            # Если фото слишком широкое (горизонтальное), подгоняем по высоте
            clip = clip.resized(height=VIDEO_H)
        else:
            # Если фото узкое, подгоняем по ширине
            clip = clip.resized(width=VIDEO_W)

        # Обрезаем лишнее по бокам или сверху/снизу, чтобы был ровный вертикальный кадр
        clip = clip.cropped(x_center=clip.w/2, y_center=clip.h/2, width=VIDEO_W, height=VIDEO_H)

        # Эффект плавного наезда (Zoom)
        def zoom_in(t):
            return 1 + ZOOM_SPEED * (t / dur)

        clip = clip.resized(zoom_in).with_position("center")
        clips.append(clip)

    # 4. Сборка финального файла
    print("Склеиваем клипы...")
    final_video = concatenate_videoclips(clips, method="compose")
    
    # Накладываем аудио
    audio_clip = AudioFileClip(audio_path).with_duration(final_video.duration)
    final_video = final_video.with_audio(audio_clip)

    # Сохраняем с НОВЫМ именем, чтобы не перепутать
    output_name = "STYLISH_REELS.mp4"
    print(f"Рендеринг в {output_name}...")
    final_video.write_videofile(output_name, fps=30, codec="libx264", audio_codec="aac")
    
    print("\n--- ГОТОВО! ---")
    print(f"Ищи файл: {output_name}")

if __name__ == "__main__":
    create_pure_vertical_video()