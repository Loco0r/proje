from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2

# Modelin tabanı
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Eğitim verisini hazırlayın
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    'C:/Users/recep/Desktop/Bird_classifier/dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Sınıf sayısını alın
num_classes = train_data.num_classes

# Modeli oluşturun
model = Sequential([
    base_model,
    Flatten(),
    Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accurary'])

history=model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    steps_per_epoch=train_data.samples // 32,
    validation_steps=test_data.samples // 32
)

model.save('C:/Users/recep/Desktop/Bird_classifier/backend/saved_model/bird_classifier.h5')