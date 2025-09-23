import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D

tf.get_logger().setLevel('ERROR')


class KneePipelineEnhanced:
    def __init__(self, yolo_model, main_model, clip_limit=2.0,
                 confidence_threshold=0.7, alpha=0.5, verbose=True):
        self.yolo_model = yolo_model
        self.main_model = main_model
        self.clip_limit = clip_limit
        self.confidence_threshold = confidence_threshold
        self.labels = ["Healthy Knee", "Osteopenia", "Osteoporosis"]
        self.alpha = alpha
        self.verbose = verbose

    # -------- CLAHE --------
    def apply_clahe(self, img):
        if len(img.shape) == 3 and img.shape[-1] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB if img.shape[2] == 3 else cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2GRAY)
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            return clahe.apply(img)
        try:
            bgr = img[:, :, :3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            return clahe.apply(gray)
        except:
            return img if len(img.shape) == 2 else cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

    # -------- YOLO --------
    def run_yolo(self, img):
        try:
            results = self.yolo_model(img)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []
            return [b for b in boxes if float(b.conf[0]) >= self.confidence_threshold]
        except Exception as e:
            if self.verbose:
                print("run_yolo error:", e)
            return []

    # -------- حساب التداخل (IoU) --------
    def compute_iou(self, boxA, boxB, margin=0):
        xA = max(boxA[0] - margin, boxB[0] - margin)
        yA = max(boxA[1] - margin, boxB[1] - margin)
        xB = min(boxA[2] + margin, boxB[2] + margin)
        yB = min(boxA[3] + margin, boxB[3] + margin)

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea

        return interArea / float(unionArea)

    # -------- فلترة البوكسات المتداخلة --------
    def filter_overlapping_boxes(self, boxes, margin=10, iou_threshold=0.1):
        if not boxes:
            return []

        boxes_sorted = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)
        final_boxes = []

        for i, b in enumerate(boxes_sorted):
            keep = True
            b_coords = b.xyxy[0].cpu().numpy()
            for kept in final_boxes:
                kept_coords = kept.xyxy[0].cpu().numpy()
                iou = self.compute_iou(b_coords, kept_coords, margin=margin)
                if iou > iou_threshold:
                    keep = False
                    break
            if keep:
                final_boxes.append(b)

        return final_boxes

    # -------- Draw Boxes --------
    def draw_boxes(self, img, boxes):
        img_drawn = img.copy()
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (255, 0, 0), 3)
        return img_drawn

    # -------- Crop to square around box (safe) --------
    def crop_from_box(self, image, box, vertical_expand=0.0, horizontal_expand=0.0, make_square=False,
                      target_size=(224, 224)):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box

        box_w, box_h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - horizontal_expand * box_w))
        x2 = min(w, int(x2 + horizontal_expand * box_w))
        y1 = max(0, int(y1 - vertical_expand * box_h))
        y2 = min(h, int(y2 + vertical_expand * box_h))

        cropped = image[y1:y2, x1:x2]

        if make_square:
            ch, cw = cropped.shape[:2]
            target_w, target_h = target_size
            diff_w = max(0, target_w - cw)
            diff_h = max(0, target_h - ch)
            left_pad = diff_w // 2
            right_pad = diff_w - left_pad
            top_pad = diff_h // 2
            bottom_pad = diff_h - top_pad
            x1 = max(0, x1 - left_pad)
            x2 = min(w, x2 + right_pad)
            y1 = max(0, y1 - top_pad)
            y2 = min(h, y2 + bottom_pad)
            cropped = image[y1:y2, x1:x2]

        return cropped

    # -------- Resize (safe) --------
    def resize_img(self, img, target_size=(224, 224)):
        target_h, target_w = target_size
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if self.verbose:
            print("resize_img: before->", img.shape, "after->", resized.shape)
        return resized

    # -------- Prepare Input --------
    def prepare_input(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img = cv2.resize(img, (224, 224))
        arr = img.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return np.array(arr, dtype=np.float32)

    # -------- Grad-CAM++ --------
    def gradcam(self, img_batch, class_idx=None):
        debug_info = {}
        if isinstance(img_batch, list):
            debug_info["warning"] = "img_batch was a list; converting to np.array"
            img_batch = np.array(img_batch, dtype=np.float32)
        elif not isinstance(img_batch, np.ndarray):
            debug_info["warning"] = f"img_batch type {type(img_batch)}; converting to np.array"
            img_batch = np.array(img_batch, dtype=np.float32)
        if img_batch.ndim != 4:
            debug_info["warning_shape_fix"] = f"input was {img_batch.shape}, expanding dims"
            img_batch = np.expand_dims(img_batch, axis=0)
        debug_info["input_type"] = type(img_batch)
        debug_info["input_shape_before"] = img_batch.shape

        last_conv_name = None
        for layer in reversed(self.main_model.layers):
            if isinstance(layer, Conv2D):
                last_conv_name = layer.name
                break
        if last_conv_name is None:
            debug_info["error"] = "No Conv2D layer found in model for Grad-CAM++"
            return None, debug_info

        try:
            grad_model = Model(
                inputs=self.main_model.input,
                outputs=[self.main_model.get_layer(last_conv_name).output, self.main_model.output]
            )
            input_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape1:
                with tf.GradientTape(persistent=True) as tape2:
                    with tf.GradientTape() as tape3:
                        conv_outputs, predictions = grad_model(input_tensor)
                        if isinstance(conv_outputs, list):
                            conv_outputs = conv_outputs[0]
                        if isinstance(predictions, list):
                            predictions = predictions[0]
                        if class_idx is None:
                            class_idx = tf.argmax(predictions[0])
                        loss = predictions[:, class_idx]

                    grads = tape3.gradient(loss, conv_outputs)
                grads2 = tape2.gradient(grads, conv_outputs)
            grads3 = tape1.gradient(grads2, conv_outputs)

            conv_outputs = conv_outputs[0].numpy()
            grads = grads[0].numpy()
            grads2 = grads2[0].numpy()
            grads3 = grads3[0].numpy()

            numerator = grads2
            denominator = 2.0 * grads2 + grads3 * conv_outputs
            denominator[denominator == 0] = 1e-8
            alphas = numerator / denominator

            weights = np.sum(alphas * np.maximum(grads, 0), axis=(0, 1))
            cam = np.sum(weights * conv_outputs, axis=-1)

            hot_map = np.maximum(cam, 0)
            cold_map = np.maximum(-cam, 0)
            hot_map /= (np.max(hot_map) + 1e-8)
            cold_map /= (np.max(cold_map) + 1e-8)
            hot_map = np.power(hot_map, 0.7)
            cold_map = np.power(cold_map, 0.7)

            H, W = img_batch.shape[1], img_batch.shape[2]
            hot_resized = cv2.resize(hot_map, (W, H))
            cold_resized = cv2.resize(cold_map, (W, H))
            hot_resized = cv2.GaussianBlur(hot_resized, (7, 7), 0)
            cold_resized = cv2.GaussianBlur(cold_resized, (7, 7), 0)

            def to_color(map_array, hot=True):
                map_uint8 = np.uint8(255 * map_array)
                color_img = np.zeros((H, W, 3), dtype=np.uint8)
                if hot:
                    color_img[:, :, 2] = map_uint8
                else:
                    color_img[:, :, 0] = map_uint8
                return color_img

            hot_color = to_color(hot_resized, hot=True)
            cold_color = to_color(cold_resized, hot=False)

            base_img = (img_batch[0] * 255).astype(np.uint8)
            if base_img.shape[-1] == 3:
                base_for_overlay = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
            else:
                base_for_overlay = base_img

            overlay_hot = cv2.addWeighted(base_for_overlay, 0.6, hot_color, 0.8, 0)
            overlay_dual = cv2.addWeighted(overlay_hot, 0.7, cold_color, 0.5, 0)

            debug_info["final_overlay_shape"] = overlay_dual.shape
            return cv2.cvtColor(overlay_dual, cv2.COLOR_BGR2RGB), debug_info

        except Exception as e:
            debug_info["error"] = str(e)
            return None, debug_info

    # -------- Process Image (main) --------
    def process(self, img):
        output = {"original": img, "clahe": None, "yolo_annotated": None, "knees": [], "warnings": []}

        img_for_clahe = img if len(img.shape) == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        output["clahe"] = self.apply_clahe(img_for_clahe)
        rgb_clahe = cv2.cvtColor(output["clahe"], cv2.COLOR_GRAY2RGB)

        boxes = self.run_yolo(rgb_clahe)
        boxes = self.filter_overlapping_boxes(boxes, margin=15, iou_threshold=0.05)  # ← تعديل التداخل
        boxes = sorted(boxes, key=lambda b: float(b.xyxy[0].cpu().numpy()[0])) if boxes else []
        output["yolo_annotated"] = self.draw_boxes(rgb_clahe, boxes)

        if self.verbose:
            print("Found boxes:", len(boxes))

        for idx, b in enumerate(boxes):
            coords = b.xyxy[0].cpu().numpy()
            if self.verbose:
                print(f"Box {idx+1} coords (x1,y1,x2,y2): {coords}")

            cropped_square = self.crop_from_box(output["clahe"], coords,
                                                vertical_expand=0.5, horizontal_expand=0.1, make_square=True)
            resized = self.resize_img(cropped_square, target_size=(224, 224))
            input_tensor = self.prepare_input(resized)

            if self.verbose:
                print(f"Knee {idx+1}: cropped {cropped_square.shape} -> resized {resized.shape} -> tensor {input_tensor.shape}")

            preds = self.main_model.predict(input_tensor, verbose=0)[0]
            class_idx = int(np.argmax(preds))
            pred_label = self.labels[class_idx]
            confidence = round(float(preds[class_idx]) * 100, 2)
            class_probs = {self.labels[i]: round(float(preds[i] * 100), 2) for i in range(len(self.labels))}

            gradcam_img, gradcam_debug = self.gradcam(input_tensor, class_idx)
            if gradcam_img is None:
                output["warnings"].append(f"Grad-CAM failed for knee {idx + 1}: {gradcam_debug.get('error', '')}")
            output["knees"].append({
                "knee_position": f"Knee {idx + 1}",
                "cropped": cropped_square,
                "resized": resized,
                "gradcam": gradcam_img,
                "gradcam_debug": gradcam_debug,
                "prediction": pred_label,
                "confidence_percent": confidence,
                "class_probabilities_percent": class_probs
            })

        return output
