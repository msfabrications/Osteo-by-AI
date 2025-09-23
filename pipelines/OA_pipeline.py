import cv2
from collections import deque

class KneeProcessor:
    def __init__(self, model, class_id=0, conf_threshold=0.65, overlap_threshold=0.0, knee_distance_threshold_ratio=0.25):
        self.model = model
        self.class_id = class_id
        self.conf_threshold = conf_threshold
        self.overlap_threshold = overlap_threshold
        self.knee_distance_threshold_ratio = knee_distance_threshold_ratio

        self.recent_originals = deque(maxlen=1)
        self.recent_clahe = deque(maxlen=1)
        self.recent_yolo_annotated = deque(maxlen=1)
        self.recent_cropped = deque(maxlen=2)
        self.recent_resized = deque(maxlen=2)

        self.skipped_images = []
        self.failed_images = []
        self.two_joint_images = []

    def reset(self):
        """مسح كل البيانات المخزنة"""
        self.recent_originals.clear()
        self.recent_clahe.clear()
        self.recent_yolo_annotated.clear()
        self.recent_cropped.clear()
        self.recent_resized.clear()
        self.skipped_images.clear()
        self.failed_images.clear()
        self.two_joint_images.clear()

    def apply_clahe(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def to_3channels(self, img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def resize_with_padding(self, image, target_size=(224, 224)):
        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        top_pad = (target_size[0] - resized.shape[0]) // 2
        bottom_pad = target_size[0] - resized.shape[0] - top_pad
        left_pad = (target_size[1] - resized.shape[1]) // 2
        right_pad = target_size[1] - resized.shape[1] - left_pad

        return cv2.copyMakeBorder(resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REPLICATE)

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
        xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def filter_non_overlapping_top2(self, boxes, scores):
        valid = [(box, score) for box, score in zip(boxes, scores) if score >= self.conf_threshold]
        valid.sort(key=lambda x: x[1], reverse=True)

        selected = []
        for box, score in valid:
            if all(self.iou(box, s[0]) <= self.overlap_threshold for s in selected):
                selected.append((box, score))
            if len(selected) == 2:
                break
        return [b for b, _ in selected]

    def crop_around_joint(self, img, box, target_size=(224, 224), margin=20, vertical_extra=20):
        x1, y1, x2, y2 = map(int, box)
        h, w = img.shape[:2]

        y1 = max(y1 - vertical_extra, 0)
        y2 = min(y2 + vertical_extra, h)
        x1 = max(x1 - margin, 0)
        x2 = min(x2 + margin, w)

        cropped = img[y1:y2, x1:x2]
        resized = self.resize_with_padding(cropped, target_size)
        return cropped, resized

    def crop_knee_joint_yolo(self, img_path, target_size=(224, 224), margin=10):
        img = cv2.imread(img_path)
        if img is None:
            return {"error": f"Failed to load the image: {img_path}"}, None

        img_clahe = self.apply_clahe(img)
        img_clahe_3ch = self.to_3channels(img_clahe)

        results = self.model(img_clahe_3ch, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        boxes = []
        scores = []
        for det in detections:
            if int(det[5]) == self.class_id:
                boxes.append(det[:4])
                scores.append(float(det[4]))

        selected_boxes = self.filter_non_overlapping_top2(boxes, scores)

        # إذا لم يتم اكتشاف المفصل
        if len(selected_boxes) == 0:
            self.failed_images.append(img_clahe_3ch.copy())
            self.skipped_images.append(img_path)
            return {"error": "No knee joint detected."}, None

        # حفظ الصور فقط بعد اكتشاف المفصل
        self.recent_originals.append(img.copy())
        self.recent_clahe.append(img_clahe.copy())

        cropped_joints = []
        annotated = img_clahe_3ch.copy()
        for box in selected_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, "Joint Detected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        self.recent_yolo_annotated.append(annotated)

        if len(selected_boxes) == 2:
            x_centers = [((box[0] + box[2]) / 2) for box in selected_boxes]
            x_centers.sort()
            min_dist = abs(x_centers[1] - x_centers[0])
            img_width = img_clahe.shape[1]
            if min_dist / img_width < self.knee_distance_threshold_ratio:
                self.failed_images.append(img_clahe_3ch.copy())
                return {"error": "Two joints detected but too close."}, None
            self.two_joint_images.append(img_clahe_3ch.copy())

        for box in selected_boxes:
            cropped, resized = self.crop_around_joint(img_clahe_3ch, box, target_size, margin)
            cropped_joints.append(resized)
            self.recent_cropped.append(cropped)
            self.recent_resized.append(resized)

        return cropped_joints, None
