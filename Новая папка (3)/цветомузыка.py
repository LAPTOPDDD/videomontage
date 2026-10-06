import os
import librosa
import numpy as np
from moviepy import ImageClip, VideoClip, CompositeVideoClip, AudioFileClip

def create_color_music():
    # 1. Поиск файлов
    images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    audio_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp3', '.wav'))]

    if not images or not audio_files:
        print("Ошибка: нужны фото и mp3 в папке!")
        return

    audio_path = audio_files[0]
    img_path = images[0]

    print(f"Анализируем частоты: {audio_path}...")
    # Загружаем аудио
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    # 2. Извлекаем частоты (STFT - преобразование Фурье)
    S = np.abs(librosa.stft(y))
    # Разделяем на 3 полосы:
    bass_data = np.mean(S[:20, :], axis=0)    # Низкие (Красный)
    mids_data = np.mean(S[20:100, :], axis=0)  # Средние (Зеленый)
    highs_data = np.mean(S[100:, :], axis=0)   # Высокие (Синий)

    # Функция нормализации (чтобы значения были от 0 до 255)
    def normalize(arr, power=1.5):
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        return np.power(arr, power) * 255 # power усиливает вспышки

    bass = normalize(bass_data)
    mids = normalize(mids_data)
    highs = normalize(highs_data)
    
    # Временная шкала для синхронизации
    times = np.linspace(0, duration, len(bass))

    # 3. Функция отрисовки кадра цветомузыки
    def make_frame(t):
        # Находим индекс массива для текущей секунды t
        idx = np.searchsorted(times, t)
        if idx >= len(bass): idx = len(bass) - 1

        # Формируем цвет RGB на основе частот
        # Бас бьет красным, середина зеленым, верха синим
        r = int(bass[idx])
        g = int(mids[idx] * 0.7) # чуть приглушим середину
        b = int(highs[idx])

        # Создаем заливку цветом на весь экран
        frame = np.full((1080, 1920, 3), [r, g, b], dtype='uint8')
        return frame

    print("Создаем видео слои...")
    # Слой 1: Динамическая цветомузыка
    color_bg = VideoClip(make_frame, duration=duration)

    # Слой 2: Ваше фото (сделаем его чуть меньше, чтобы сзади была "аура" цвета)
    photo = (ImageClip(img_path)
             .with_duration(duration)
             .resized(height=900) # оставляем место для свечения по краям
             .with_position("center"))

    # 4. Сборка и сохранение
    final_video = CompositeVideoClip([color_bg, photo])
    audio_clip = AudioFileClip(audio_path)
    final_video = final_video.with_audio(audio_clip)

    print("Начинаем рендеринг цветомузыки (это быстро)...")
    final_video.write_videofile("color_music_show.mp4", fps=24, codec="libx264")
    print("Готово! Проверяй файл color_music_show.mp4")

if __name__ == "__main__":
    create_color_music()