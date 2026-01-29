from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def track(self, frame):
        results = self.model.track(frame, persist=True, verbose=False, classes=[2, 5, 7])
        return results