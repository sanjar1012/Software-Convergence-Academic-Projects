import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2

# 1. Build a Custom Object Detection Head on top of MobileNetV2
def build_traffic_detector(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze backbone for lightweight edge deployment
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Dual output heads: 1 for bounding box regression, 1 for vehicle class probability
    class_output = layers.Dense(1, activation='sigmoid', name='class_output')(x) # Vehicle vs No Vehicle
    bbox_output = layers.Dense(4, activation='linear', name='bbox_output')(x)   # [xmin, ymin, xmax, ymax]
    
    model = models.Model(inputs=base_model.input, outputs=[class_output, bbox_output])
    return model

# 2. Compile Model with Multi-task Loss
model = build_traffic_detector()
model.compile(
    optimizer='adam',
    loss={'class_output': 'binary_crossentropy', 'bbox_output': 'mse'},
    metrics={'class_output': 'accuracy'}
)

# 3. Simulate Video Frame Processing and Traffic Density Calculation
def process_traffic_video(video_path, model, threshold=0.6):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("Starting real-time traffic analysis pipeline...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Preprocess frame to match model input dimensions
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        input_tensor = np.expand_dims(normalized_frame, axis=0)
        
        # Inference
        class_pred, bbox_pred = model.predict(input_tensor, verbose=0)
        
        # Analytics: Calculate artificial vehicle density metrics for the CV description
        vehicle_detected = class_pred[0][0] > threshold
        
        if frame_count % 30 == 0:  # Log status every 30 frames (~1 second of video)
            status = "Heavy Traffic / Potential Congestion" if vehicle_detected else "Clear"
            print(f"Frame {frame_count} | Detection Confidence: {class_pred[0][0]:.2f} | Status: {status}")
            
        # Break loop after simulating 150 frames for demonstration purposes
        if frame_count >= 150:
            break
            
    cap.release()
    print("Pipeline processing completed successfully.")

if __name__ == '__main__':
    # Initialize model summary to verify architecture
    model.summary()
    print("\n[Mock Run] Simulating pipeline with randomized feed...")
    # Generate mock frame to test pipeline execution
    mock_video_feed = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mock_input = np.expand_dims(mock_video_feed / 255.0, axis=0)
    conf, bbox = model.predict(mock_input)
    print(f"Mock Inference Successful. Detected confidence score: {conf[0][0]:.4f}")
