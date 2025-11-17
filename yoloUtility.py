import numpy as np
from ultralytics import YOLO


def detect_boxes(image, model_path='yolov8n.pt', conf_threshold=0.25):
    """
    Detect objects in an image using YOLO and return bounding boxes for SAM pipeline.
    
    Args:
        image: Input image as numpy array (BGR or RGB format)
        model_path: Path to YOLO model weights file (default: 'yolov8n.pt')
                   Can be 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
                   or a custom trained model path
        conf_threshold: Confidence threshold for detections (default: 0.25)
    
    Returns:
        List of bounding boxes in format [x1, y1, x2, y2] for each detection
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image, conf=conf_threshold)
    
    # Extract bounding boxes
    boxes = []
    for result in results:
        # result.boxes contains the bounding boxes
        if result.boxes is not None:
            # Get boxes in xyxy format (x1, y1, x2, y2)
            xyxy_boxes = result.boxes.xyxy.cpu().numpy()
            
            # Convert to list of lists [x1, y1, x2, y2]
            for box in xyxy_boxes:
                boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
    
    return boxes

