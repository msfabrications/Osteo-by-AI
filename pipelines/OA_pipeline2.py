import cv2
import numpy as np
import tensorflow as tf
import keras
import base64

class Pipeline2:
    def __init__(self, model_path):
        try:
            self.model = keras.models.load_model(model_path)
            print(f"[Pipeline2] Model loaded from: {model_path}")
        except Exception as e:
            print(f"[Pipeline2] Failed to load model: {e}")
            self.model = None

    def _get_last_conv_layer(self):
        if not self.model:
            return None
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        return None

    @staticmethod
    def _gradcam_to_base64(img):
        if img is None:
            return None
        try:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return None

    def predict_knee_image(self, resized_img, return_full=False):
        debug_info = []
        gradcam_base64 = None
        class_labels = {0: "Normal", 1: "Mild OA", 2: "Moderate OA", 3: "Severe OA"}

        # --- RGB ---
        if len(resized_img.shape) == 2 or resized_img.shape[2] == 1:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
            debug_info.append("Converted to 3-channel RGB.")
        else:
            debug_info.append("Image already 3-channel RGB.")

        # --- preprocessing ---
        input_img = np.expand_dims(resized_img, axis=0).astype("float32") / 255.0
        debug_info.append(f"Image preprocessing successful. Shape: {input_img.shape}")

        # --- prediction ---
        pred = self.model.predict(input_img, verbose=0)
        predicted_class = int(np.argmax(pred))
        confidence = float(pred[0][predicted_class] * 100)
        debug_info.append(f"Prediction parsing successful: {class_labels[predicted_class]} ({confidence:.2f}%)")

        # --- Grad-CAM++ ---
        last_conv_layer_name = "block6d_project_conv"
        try:
            grad_model = keras.models.Model(
                self.model.inputs,
                [self.model.get_layer(last_conv_layer_name).output, self.model.output]
            )
            debug_info.append(f"Grad model built successfully. Last conv layer: {last_conv_layer_name}")

            input_tensor = tf.convert_to_tensor(input_img, dtype=tf.float32)

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_tensor)
                if isinstance(conv_outputs, list):
                    conv_outputs = conv_outputs[0]
                if isinstance(predictions, list):
                    predictions = predictions[0]

                loss = predictions[:, predicted_class]

            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                debug_info.append("[Grad-CAM] grads is None → using original image")
                gradcam_base64 = self._gradcam_to_base64(resized_img.astype(np.uint8))
            else:
                conv_outputs = conv_outputs[0].numpy()
                grads = grads[0].numpy()
                debug_info.append(f"Conv outputs shape: {conv_outputs.shape}, Grads shape: {grads.shape}")

                # --- Grad-CAM++ core ---
                weights = np.mean((grads ** 2) * np.maximum(conv_outputs, 0), axis=(0, 1))
                conv_outputs_weighted = conv_outputs * weights[np.newaxis, np.newaxis, :]
                heatmap = np.mean(conv_outputs_weighted, axis=-1)
                heatmap = np.maximum(heatmap, 0)

                if np.max(heatmap) > 0:
                    heatmap /= np.max(heatmap)

                    h, w = resized_img.shape[:2]
                    heatmap = cv2.resize(heatmap, (w, h))
                    heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
                    heatmap = np.power(heatmap, 0.8)

                    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

                    if len(resized_img.shape) == 2 or resized_img.shape[2] == 1:
                        resized_img_color = cv2.cvtColor(resized_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    else:
                        resized_img_color = resized_img.astype(np.uint8)

                    overlay = cv2.addWeighted(resized_img_color, 0.6, heatmap_color, 0.4, 0)
                    gradcam_base64 = self._gradcam_to_base64(overlay)

                    debug_info.append("[Grad-CAM++] Overlay applied with smooth colors and correct alignment")
                else:
                    gradcam_base64 = self._gradcam_to_base64(resized_img.astype(np.uint8))
                    debug_info.append("[Grad-CAM++] Heatmap empty → replaced with original image")

        except Exception as e:
            gradcam_base64 = self._gradcam_to_base64(resized_img.astype(np.uint8))
            debug_info.append(f"[Grad-CAM++] Failed: {e} → replaced with original image")

        if return_full:
            return pred, gradcam_base64, debug_info
        else:
            return f"{class_labels[predicted_class]} ({confidence:.2f}%)", gradcam_base64, debug_info







