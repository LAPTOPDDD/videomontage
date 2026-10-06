import ollama
import os

def get_photo_description(image_path):
    print(f"AI анализирует: {image_path}...")
    # Используем легкую vision-модель moondream
    response = ollama.generate(
        model='moondream',
        prompt='Опиши коротко в 3-5 словах, что на этом фото для заголовка видео',
        images=[image_path]
    )
    return response['response']

# Пример того, как это встроить в наш прошлый скрипт
images = [f for f in os.listdir('.') if f.endswith('.jpg')]
for img in images:
    description = get_photo_description(img)
    print(f"Для фото {img} нейронка придумала: {description}")