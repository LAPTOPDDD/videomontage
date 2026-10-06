import os
import librosa
import numpy as np
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip

def create_pulsing_video():
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav'))]

    if not images or not audio_files: return
    audio_path = audio_files[0]

    # 1. Находим биты
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Берем каждый 2-й бит для смены фото
    step = 2
    selected_beats = beat_times[::step]
    durations = np.diff(selected_beats)

    clips = []
    num_frames = min(len(images), len(durations))

    for i in range(num_frames):
        dur = durations[i]
        
        # Создаем функцию для эффекта пульсации
        # t - это время внутри текущего кусочка видео (от 0 до dur)
        # Мы заставляем размер меняться от 1.0 до 1.1 и обратно
        def zoom(t):
            # Математика: создаем "горбик" на каждом ударе
            # sin даст плавность, можно настроить силу пульсации
            pulse = 1 + 0.05 * np.sin(np.pi * (t / dur)) 
            return pulse

        clip = (ImageClip(images[i])
                .with_duration(dur)
                .resized(height=1080)
                .resized(zoom)) # Применяем динамическое изменение размера
        
        clips.append(clip)

    final_video = concatenate_videoclips(clips, method="compose")
    audio_clip = AudioFileClip(audio_path).with_duration(final_video.duration)
    final_video = final_video.with_audio(audio_clip)

    final_video.write_videofile("pulse_video.mp4", fps=24, codec="libx264")

if __name__ == "__main__":
    create_pulsing_video()