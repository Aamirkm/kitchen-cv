from ultralytics import YOLO

# Load a pre-trained YOLOv11 model. 
# We use a pre-trained model so it already understands basic shapes and objects.
# This process, called "transfer learning," is much faster and more effective than training from scratch.
model = YOLO('yolo11n.pt')

# --- START TRAINING ---
# The 'data' argument must point to the 'data.yaml' file inside the folder you downloaded from Roboflow.
# Make sure you update the path if your folder has a different name.
if __name__ == '__main__':
    results = model.train(data='labeled_dataset/data.yaml',  # UPDATE THIS PATH
                          epochs=100,                        # Number of times to train on the whole dataset
                          imgsz=640)                         # Image size for training

    print("Training complete! Your new model is saved in the 'runs' folder.")
