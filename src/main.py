import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0, InceptionV3, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report

# Define paths
dataset_dir = "original_dataset"  # Update this path to your dataset directory
categories = ["bacteria blight", "curl Virus", "healthy leaves", "herbicide growth damage", 
              "leafe hopper jassids", "leafe Redding", "leafe variegation", "invalid"]  # Add "invalid" class

# Image dimensions
img_width, img_height = 224, 224
batch_size = 32
num_classes = len(categories)

# Data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load dataset
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Function to build a transfer learning model
def build_model(base_model, num_classes):
    base_model.trainable = False  # Freeze the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# List of pre-trained models to compare
models = {
    "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3)),
    "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3)),
    "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3)),
    "InceptionV3": InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
}

# Train and evaluate each model
results = {}
for model_name, base_model in models.items():
    print(f"Training {model_name}...")
    
    # Build the model
    model = build_model(base_model, num_classes)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        f"{model_name}_best_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=20,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(validation_generator)
    results[model_name] = val_accuracy
    print(f"{model_name} Validation Accuracy: {val_accuracy * 100:.2f}%")

# Compare results
print("\nModel Comparison:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy * 100:.2f}%")

# Select the best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%")

# Save the best model
best_model = tf.keras.models.load_model(f"{best_model_name}_best_model.h5")
best_model.save("best_cotton_disease_model.h5")
print("Best model saved as 'best_cotton_disease_model.h5'.")

# Test the model on a random image (e.g., sky image)
def predict_image(image_path, model, threshold=0.9):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    if confidence < threshold:
        return "Invalid image (not a cotton leaf)"
    else:
        return categories[predicted_class]

# Test with a random image
random_image_path = "path_to_random_image.jpg"  # Replace with the path to a random image (e.g., sky)
result = predict_image(random_image_path, best_model)
print(f"Prediction for random image: {result}")