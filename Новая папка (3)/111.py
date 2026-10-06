import os
import librosa
import numpy as np

# Импорты для MoviePy 2.0+
try:
    from moviepy import ImageClip, concatenate_videoclips, AudioFileClip
except ImportError:
    from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

def create_video_to_beat():
    # 1. Поиск файлов
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav'))]

    if not images:
        print("Ошибка: Фотографии не найдены!")
        return
    if not audio_files:
        print("Ошибка: Аудио файл не найден!")
        return

    audio_path = audio_files[0]
    print(f"Используем аудио: {audio_path}")
    print(f"Найдено фото: {len(images)}")

    # 2. Анализ музыки
    print("Анализируем музыку (ищем биты)...")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Вычисляем длительность между битами
    durations = np.diff(beat_times)
    
    # 3. Сборка клипов
    clips = []
    num_frames = min(len(images), len(durations))
    
    print(f"Собираем видео из {num_frames} кадров...")
    for i in range(num_frames):
        # В MoviePy 2.0 используем .with_duration() и .resized()
        img_clip = (ImageClip(images[i])
                    .with_duration(durations[i])
                    .resized(height=1080)) # подгоняем под одну высоту
        clips.append(img_clip)

    # 4. Склеиваем и добавляем звук
    final_video = concatenate_videoclips(clips, method="compose")
    
    # Накладываем аудио
    audio_clip = AudioFileClip(audio_path).with_duration(final_video.duration)
    final_video = final_video.with_audio(audio_clip)

    # 5. Сохранение
    output_filename = "result_video_sync.mp4"
    print("Начинаем рендеринг...")
    # Задаем fps для итогового видео
    final_video.write_videofile(output_filename, fps=24, codec="libx264", audio_codec="aac")
    print(f"Готово! Файл: {output_filename}")

if __name__ == "__main__":
    create_video_to_beat()