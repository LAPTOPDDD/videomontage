import os
import librosa
import numpy as np
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip, VideoClip, CompositeVideoClip

# ================= НАСТРОЙКИ =================
BEATS_STEP = 4          # Смена фото каждые N ударов
BARS_COUNT = 30         # Количество столбиков визуализатора
VIDEO_HEIGHT = 1080
VIDEO_WIDTH = 1920
# =============================================

def create_video_with_visualizer():
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav'))]

    if not images or not audio_files:
        print("Файлы не найдены!")
        return

    audio_path = audio_files[0]
    
    # 1. Анализ тактов для смены фото
    print("Анализируем темп...")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    selected_beats = beat_times[::BEATS_STEP]
    durations = np.diff(selected_beats)

    # 2. Анализ частот для визуализатора (спектрограмма)
    print("Готовим визуализацию звука...")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=BARS_COUNT)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Нормализуем значения от 0 до 1
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())

    # 3. Сборка слайдшоу
    clips = []
    num_frames = min(len(images), len(durations))
    for i in range(num_frames):
        img = ImageClip(images[i]).with_duration(durations[i]).resized(height=VIDEO_HEIGHT)
        # Центрируем фото, если оно шире экрана
        if img.w > VIDEO_WIDTH:
            img = img.cropped(x_center=img.w/2, width=VIDEO_WIDTH)
        clips.append(img)
    
    slideshow = concatenate_videoclips(clips, method="compose")
    total_duration = slideshow.duration

    # 4. Создание функции рисования визуализатора
    def make_frame(t):
        # Создаем пустой прозрачный кадр
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype="uint8")
        
        # Индекс данных спектрограммы для текущего момента времени t
        idx = int((t / total_duration) * (S_db.shape[1] - 1))
        current_spec = S_db[:, idx]

        bar_w = VIDEO_WIDTH // BARS_COUNT
        for i in range(BARS_COUNT):
            # Высота столбика (макс 300 пикселей)
            bar_h = int(current_spec[i] * 300)
            
            # Рисуем прямоугольник (столбик)
            x1, y1 = i * bar_w + 5, VIDEO_HEIGHT - 50
            x2, y2 = (i + 1) * bar_w - 5, VIDEO_HEIGHT - 50 - bar_h
            
            # Цвет столбика (белый с небольшим градиентом)
            frame[y2:y1, x1:x2] = [255, 255, 255] 
            
        return frame

    # Создаем клип с анимацией столбиков
    viz_clip = VideoClip(make_frame, duration=total_duration).with_mask()

    # 5. Накладываем визуализатор на слайдшоу
    final_video = CompositeVideoClip([slideshow, viz_clip.with_opacity(0.7)])
    
    audio_clip = AudioFileClip(audio_path).with_duration(total_duration)
    final_video = final_video.with_audio(audio_clip)

    print("Рендеринг (с визуализацией это дольше)...")
    final_video.write_videofile("video_with_viz.mp4", fps=24, codec="libx264")

if __name__ == "__main__":
    create_video_with_visualizer()