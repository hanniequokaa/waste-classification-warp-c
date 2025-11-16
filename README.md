**Edge AI Prototype — Waste Classification (WaRP-C Dataset)
Project Overview**

This project demonstrates an Edge AI application for classifying recyclable waste using a lightweight CNN model. The goal is to show how a model can be trained, converted to TensorFlow Lite (TFLite), and prepared for deployment on edge devices like Raspberry Pi or microcontrollers.

**Dataset
**
**Name: WaRP-C (WaRP — Waste Recycling Plant Dataset)**

**Source:** Kaggle

**Content:**

train_crops/ — 8,823 training images

test_crops/ — 1,583 testing images

28 recyclable waste classes (plastic, glass, cardboard, detergents, cans)

**Purpose**: Classification task

**Model Architecture**

A lightweight CNN was used for Edge AI deployment:

Conv2D(16) → MaxPooling2D
Conv2D(32) → MaxPooling2D
Flatten
Dense(64, ReLU)
Dense(28, Softmax)


Input shape: 128x128 RGB images

Loss function: Sparse Categorical Crossentropy

Optimizer: Adam

**Training Details**

Epochs: 5 (can be increased for higher accuracy)

Batch size: 32

Validation accuracy: 0.7421

Data preprocessing: Images normalized to 0–1

**TensorFlow Lite Conversion**

The trained model was converted to TFLite for edge deployment:

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("warp_c_model.tflite", "wb") as f:
    f.write(tflite_model)


File: warp_c_model.tflite

Optimized for smaller size and faster inference on edge devices.

Sample Inference Using TFLite
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="warp_c_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example image from validation set
img = val_images[0]  # preprocessed 128x128 image
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
interpreter.invoke()
pred = interpreter.get_tensor(output_details[0]['index'])
pred_label = np.argmax(pred[0])


True vs Predicted labels can be compared to check accuracy.

**Deployment Notes**

Copy warp_c_model.tflite to your target device (e.g., Raspberry Pi).

Install tflite-runtime or TensorFlow.

Run inference using Python or integrate into microcontrollers using TFLite Micro.

Suitable for real-time waste sorting applications.

References

WaRP Dataset: Kaggle Dataset

TensorFlow Lite Documentation: https://www.tensorflow.org/lite
# waste-classification-warp-c
