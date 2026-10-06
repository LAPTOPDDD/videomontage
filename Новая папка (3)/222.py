import os
import librosa
import numpy as np

try:
    from moviepy import ImageClip, concatenate_videoclips, AudioFileClip
except ImportError:
    from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

# ================= НАСТРОЙКИ =================
BEATS_STEP = 16  # Каждые сколько ударов менять фото? (2, 4, 6 и т.д.)
# =============================================

def create_video_to_beat():
    files = os.listdir('.')
    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav'))]

    if not images or not audio_files:
        print("Файлы не найдены!")
        return

    audio_path = audio_files[0]
    
    print(f"Анализ музыки: {audio_path}...")
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # --- ГЛАВНОЕ ИЗМЕНЕНИЕ ТУТ ---
    # Выбираем каждый N-й удар (например, каждый 2-й, 4-й или 6-й)
    selected_beats = beat_times[::BEATS_STEP]
    
    # Считаем длительность между выбранными ударами
    durations = np.diff(selected_beats)
    # -----------------------------

    clips = []
    num_frames = min(len(images), len(durations))
    
    print(f"Смена фото каждые {BEATS_STEP} удара(ов). Итого кадров: {num_frames}")

    for i in range(num_frames):
        img_clip = (ImageClip(images[i])
                    .with_duration(durations[i])
                    .resized(height=1080))
        clips.append(img_clip)

    final_video = concatenate_videoclips(clips, method="compose")
    audio_clip = AudioFileClip(audio_path).with_duration(final_video.duration)
    final_video = final_video.with_audio(audio_clip)

    output_filename = f"video_{BEATS_STEP}_beats.mp4"
    final_video.write_videofile(output_filename, fps=24, codec="libx264", audio_codec="aac")
    print(f"Готово! Файл: {output_filename}")

if __name__ == "__main__":
    create_video_to_beat()