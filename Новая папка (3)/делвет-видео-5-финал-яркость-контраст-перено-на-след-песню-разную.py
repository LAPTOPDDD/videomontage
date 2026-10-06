import os
import librosa
import numpy as np
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# ================= НАСТРОЙКИ =================
BEATS_STEP = 2             # Смена фото каждые 2 удара
VIDEO_W, VIDEO_H = 1080, 1920 
ZOOM_SPEED = 0.12

# Настройки цветокоррекции
BRIGHTNESS = 10            # Яркость (от -100 до 100)
CONTRAST = 1.3             # Контраст (1.0 - без изменений, 1.3 - сочно, 1.5 - очень контрастно)
# =============================================

def apply_color_correction(img):
    """Ручная цветокоррекция через numpy (быстро и надежно)"""
    # Преобразуем в float для расчетов
    img = img.astype(float)
    # Применяем контраст и яркость
    img = img * CONTRAST + BRIGHTNESS
    # Ограничиваем значения от 0 до 255 и возвращаем в формат картинки
    return np.clip(img, 0, 255).astype('uint8')

def process_batch(image_batch, durations, audio_path, output_name):
    """Сборка одного видеофайла"""
    print(f"--- Сборка {output_name} под трек: {os.path.basename(audio_path)} ---")
    clips = []
    
    num_frames = min(len(image_batch), len(durations))

    for i in range(num_frames):
        dur = durations[i]
        img_path = image_batch[i]

        # 1. Загрузка и кроп
        clip = ImageClip(img_path).with_duration(dur)
        
        target_ratio = VIDEO_W / VIDEO_H
        img_ratio = clip.w / clip.h

        if img_ratio > target_ratio:
            clip = clip.resized(height=VIDEO_H)
        else:
            clip = clip.resized(width=VIDEO_W)

        clip = clip.cropped(x_center=clip.w/2, y_center=clip.h/2, width=VIDEO_W, height=VIDEO_H)

        # 2. ПРИМЕНЯЕМ ЦВЕТОКОРРЕКЦИЮ (через функцию fl_image)
        clip = clip.image_transform(apply_color_correction)

        # 3. Эффект Zoom
        def zoom_fn(t):
            return 1 + ZOOM_SPEED * (t / dur)
        
        clip = clip.resized(zoom_fn).with_position("center")
        clips.append(clip)

    if not clips:
        return

    # Собираем клип
    final_video = concatenate_videoclips(clips, method="compose")
    
    # Подрезаем аудио
    audio_clip = AudioFileClip(audio_path).with_duration(final_video.duration)
    final_video = final_video.with_audio(audio_clip)

    print(f"Рендеринг {output_name}...")
    final_video.write_videofile(output_name, fps=30, codec="libx264", audio_codec="aac")

def main():
    # Поиск файлов
    files = os.listdir('.')
    all_images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    audio_files = sorted([f for f in files if f.lower().endswith(('.mp3', '.wav'))])
    
    if not all_images or not audio_files:
        print("Ошибка: Добавьте фото и хотя бы один mp3!")
        return

    img_idx = 0
    video_counter = 1

    # Цикл по фотографиям
    while img_idx < len(all_images):
        current_audio = audio_files[(video_counter - 1) % len(audio_files)]
        
        print(f"\n>>> ГОТОВИМ ВИДЕО №{video_counter} <<<")
        print(f"Анализ ритма: {current_audio}")
        
        y, sr = librosa.load(current_audio)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        durations = np.diff(beat_times[::BEATS_STEP])
        
        capacity = len(durations)
        batch = all_images[img_idx : img_idx + capacity]
        
        if not batch:
            break

        output_filename = f"VIDEO_PART_{video_counter}.mp4"
        process_batch(batch, durations, current_audio, output_filename)
        
        img_idx += len(batch)
        video_counter += 1

    print(f"\nУСПЕХ! Всего создано роликов: {video_counter - 1}")

if __name__ == "__main__":
    main()