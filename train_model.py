import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

directory='C:/Users/recep/Desktop/Bird_classifier/backend/model_training/dataset/train'
for subdir, dirs, files in os.walk(directory):
    for file in files:
        try:
            img_path=os.path.join(subdir, file)
            with Image.open(img_path) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            print(f'Bozuk dosya:{file}')

# Veri yolları
train_dir = 'C:/Users/recep/Desktop/Bird_classifier/backend/model_training/dataset/train'
test_dir = 'C:/Users/recep/Desktop/Bird_classifier/backend/model_training/dataset/test'

# Veri işleme
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')
train_data=train_datagen.flow_from_directory('C:/Users/recep/Desktop/Bird_classifier/backend/model_training/dataset/train',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')

# Model tanımlama
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model eğitimi
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=test_data.samples // test_data.batch_size
)

# Modeli kaydet
model.save('C:/Users/recep/Desktop/Bird_classifier/backend/saved_model/bird_classifier.h5')
print("Model başarıyla kaydedildi.")
