import os
import librosa
import numpy as np
from moviepy import ImageClip, VideoClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips

# ================= НАСТРОЙКИ =================
BEATS_STEP = 2            # Каждые сколько ударов менять фото
VIDEO_W, VIDEO_H = 1920, 1080
# =============================================

def create_pure_video():
    # 1. Поиск файлов
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav'))]
    
    if not images or not audio_files:
        print("Ошибка: Добавьте фото и mp3 в папку!")
        return
    
    audio_path = audio_files[0]

    # 2. Анализ аудио (Биты и Частоты)
    print("Анализируем ритм музыки...")
    y, sr = librosa.load(audio_path)
    
    # Биты для смены фото
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    selected_beats = beat_times[::BEATS_STEP]
    durations = np.diff(selected_beats)

    # Частоты для цветомузыки (Низкие, Средние, Высокие)
    S = np.abs(librosa.stft(y))
    bass_ch = np.mean(S[:20, :], axis=0)
    mids_ch = np.mean(S[20:100, :], axis=0)
    highs_ch = np.mean(S[100:, :], axis=0)

    def norm(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return np.power(arr, 1.5) * 255

    bass = norm(bass_ch)
    mids = norm(mids_ch)
    highs = norm(highs_ch)
    color_times = np.linspace(0, librosa.get_duration(y=y, sr=sr), len(bass))

    # 3. Функция фона (Цветомузыка)
    def make_bg_frame(t):
        idx = np.searchsorted(color_times, t)
        if idx >= len(bass): idx = len(bass)-1
        # RGB пульсация
        color = [int(bass[idx]), int(mids[idx]*0.5), int(highs[idx])]
        return np.full((VIDEO_H, VIDEO_W, 3), color, dtype='uint8')

    # 4. Сборка видеоряда
    print(f"Собираем видео из {len(images)} фото...")
    clips = []
    
    # Видео закончится, когда закончатся либо фото, либо биты в музыке
    num_to_make = min(len(images), len(durations))

    for i in range(num_to_make):
        dur = durations[i]
        img_path = images[i]

        # Эффект плавного приближения (Zoom)
        def zoom(t):
            # Увеличиваем от 1.0 до 1.08 за время кадра
            return 1 + 0.08 * (t / dur)

        img_clip = (ImageClip(img_path)
                    .with_duration(dur)
                    .resized(height=VIDEO_H - 160) # фото чуть меньше экрана для "ауры"
                    .resized(zoom)
                    .with_position("center"))
        
        clips.append(img_clip)

    # 5. Финальный монтаж
    # Склеиваем фото
    slideshow = concatenate_videoclips(clips, method="compose")
    
    # Создаем фон цветомузыки на всю длину слайдшоу
    bg_layer = VideoClip(make_bg_frame, duration=slideshow.duration)
    
    # Накладываем фото на фон
    final_video = CompositeVideoClip([bg_layer, slideshow.with_position("center")])
    
    # Накладываем звук
    audio_clip = AudioFileClip(audio_path).with_duration(final_video.duration)
    final_video = final_video.with_audio(audio_clip)

    # 6. Сохранение
    output_name = "RESULT_VIDEO_PUMP.mp4"
    print("Начинаем рендеринг...")
    final_video.write_videofile(output_name, fps=24, codec="libx264", audio_codec="aac")
    print(f"\nГОТОВО! Видео сохранено: {output_name}")

if __name__ == "__main__":
    create_pure_video()