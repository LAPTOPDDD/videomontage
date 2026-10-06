import os
import ollama
import librosa
import numpy as np
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip

# Используем ОДНУ модель, чтобы диск не умирал от переключений
MODEL_NAME = 'llava' 

def ask_ai_fast(image_path):
    print(f"-> Нейронка изучает: {image_path}")
    try:
        # Просим сразу на русском, модель Llava это понимает терпимо
        res = ollama.generate(
            model=MODEL_NAME,
            prompt='Опиши это фото одним очень коротким заголовком на русском языке (2-3 слова).',
            images=[image_path]
        )
        text = res['response'].strip().replace('"', '')
        print(f"-> AI сказал: {text}")
        return text
    except Exception as e:
        print(f"Ошибка AI: {e}")
        return "Мой кадр"

def create_video():
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg'))])[:5] # Тест на 5 фото
    audio_files = [f for f in files if f.lower().endswith('.mp3')]
    
    if not images or not audio_files: return
    
    print("--- НАЧАЛО ОБРАБОТКИ ---")
    
    # Сначала соберем все описания (это самая тяжелая часть)
    descriptions = []
    for img in images:
        descriptions.append(ask_ai_fast(img))

    # Анализ музыки
    print("Анализ музыки...")
    y, sr = librosa.load(audio_files[0])
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    durations = np.diff(beat_times[::2]) # Берем каждый второй такт

    # Сборка видео
    clips = []
    for i in range(len(images)):
        dur = durations[i] if i < len(durations) else 2.0
        # Пока просто фото с зумом, без наложения текста (чтобы проверить скорость)
        clip = ImageClip(images[i]).with_duration(dur).resized(height=1080)
        clips.append(clip)

    final_video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_files[0]).with_duration(final_video.duration)
    final_video = final_video.with_audio(audio)

    print("Рендеринг видео...")
    final_video.write_videofile("test_ai_video.mp4", fps=24, codec="libx264")

if __name__ == "__main__":
    create_video()