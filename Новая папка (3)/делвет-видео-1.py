import os
import ollama
import librosa
import numpy as np
from moviepy import ImageClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip

# Настройки
BEATS_STEP = 2 
MODEL_VISION = 'moondream'
MODEL_TEXT = 'llama3:8b' # Текстовая модель для перевода и стиля

def ask_ai(image_path):
    print(f"--- Анализ {image_path} ---")
    # 1. Получаем описание на английском
    res_vision = ollama.generate(
        model=MODEL_VISION,
        prompt='Describe this image in one simple sentence focus on content.',
        images=[image_path]
    )
    english_desc = res_vision['response']
    
    # 2. Просим Llama 3 перевести на русский в стиле заголовка
    prompt_translate = f"Переведи это описание фото на русский язык. Сделай это как короткий красивый заголовок для видео (2-4 слова): '{english_desc}'"
    res_text = ollama.generate(model=MODEL_TEXT, prompt=prompt_translate)
    
    russian_desc = res_text['response'].strip().replace('"', '')
    print(f"AI выбрал заголовок: {russian_desc}")
    return russian_desc

def create_ai_video():
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:10] # возьмем первые 10 для теста
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav'))]
    
    if not images or not audio_files: return
    audio_path = audio_files[0]

    # Анализ музыки
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    selected_beats = beat_times[::BEATS_STEP]
    durations = np.diff(selected_beats)

    clips = []
    
    for i in range(min(len(images), len(durations))):
        img_path = images[i]
        dur = durations[i]
        
        # Получаем AI текст
        title_text = ask_ai(img_path)

        # Создаем клип с фото и зумом
        def zoom(t):
            return 1 + 0.04 * (t / dur)

        img_clip = (ImageClip(img_path)
                    .with_duration(dur)
                    .resized(height=1080)
                    .resized(zoom))

        # Создаем текст (белый с черной обводкой)
        # ВАЖНО: убедитесь, что путь к шрифту верный или используйте стандартный
        txt_clip = TextClip(
            text=title_text,
            font_size=70,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            size=(1920*0.8, None)
        ).with_duration(dur).with_position(('center', 800)) # текст внизу

        # Совмещаем фото и текст
        final_clip = CompositeVideoClip([img_clip, txt_clip])
        clips.append(final_clip)

    # Финальная сборка
    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path).with_duration(video.duration)
    video = video.with_audio(audio)

    print("Рендеринг финального AI-видео...")
    video.write_videofile("ai_movie_result.mp4", fps=24, codec="libx264")

if __name__ == "__main__":
    create_ai_video()