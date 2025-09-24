#import time, os
from flask import Flask,  render_template
from flask import request, jsonify
from flask_cors import CORS
import os
import time
import joblib
import base64, cv2, numpy as np, traceback
from pipelines.OA_pipeline import KneeProcessor
from ultralytics import YOLO
from keras.models import load_model
from pipelines.OA_pipeline2 import Pipeline2
from pipelines.OP_Frame1 import bmd_diagnosis
from pipelines.OP_Frame2 import KneePipelineEnhanced
from sklearn.preprocessing import LabelEncoder
#OA
pipeline2 = Pipeline2("models/EfficientNetB3_model_3_OA.keras")


app = Flask(__name__)
CORS(app)


#YOLO
yolo_model_path = "models/cv_yolo_model_last.pt"
yolo_model = YOLO(yolo_model_path)
knee_processor = KneeProcessor(yolo_model)



#OP
#main_model = load_model("models/OP_Image_model.keras")
main_model = load_model("models/best_modelEfficientNetB3 (5).keras")
pipeline = KneePipelineEnhanced(yolo_model=yolo_model, main_model=main_model)

model = joblib.load("models/osteoporosis_risk_model.pkl")


def encode_image_to_base64(img, fmt=".png"):
    """
    Converts an image to a Base64 string safely.
    Handles any exceptions and ensures dtype compatibility.

    Args:
        img (np.ndarray): Image array (H x W x C)
        fmt (str): Image format, e.g., ".png" or ".jpg"

    Returns:
        str or None: Base64-encoded image string or None if failed
    """
    if img is None:
        return None

    try:

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # تحويل الصورة إلى صيغة ضغط
        success, buffer = cv2.imencode(fmt, img)
        if not success:
            # محاولة ثانية بصيغة JPEG إذا فشل PNG
            success, buffer = cv2.imencode(".jpg", img)
            if not success:
                return None


        return f"data:image/{fmt[1:]};base64," + base64.b64encode(buffer).decode()

    except Exception as e:

        return None



def clear_pipeline_memory():
    knee_processor.recent_originals.clear()
    knee_processor.recent_clahe.clear()
    knee_processor.recent_yolo_annotated.clear()
    knee_processor.recent_cropped.clear()
    knee_processor.recent_resized.clear()
    knee_processor.skipped_images.clear()
    knee_processor.failed_images.clear()
    knee_processor.two_joint_images.clear()



@app.route('/process_oa', methods=['POST'])
def process_oa():
    if 'xray' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    files = request.files.getlist('xray')
    if not files:
        return jsonify({'error': 'No files found in request'}), 400

    final_output = []
    errors = []

    for file_index, file in enumerate(files):
        #  مسح قوائم الركبة قبل معالجة كل ملف
        knee_processor.recent_originals.clear()
        knee_processor.recent_clahe.clear()
        knee_processor.recent_yolo_annotated.clear()
        knee_processor.recent_cropped.clear()
        knee_processor.recent_resized.clear()
        knee_processor.skipped_images.clear()
        knee_processor.failed_images.clear()
        knee_processor.two_joint_images.clear()

        print(f"[Flask] Received file: {file.filename}")

        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img is None:
                errors.append(f"Failed to read the uploaded image: {file.filename}")
                continue

            ext = file.filename.split('.')[-1].lower()
            if ext not in ['png', 'jpg', 'jpeg', 'bmp', 'dcm']:
                errors.append(f"Unsupported file extension: {ext}")
                continue

            temp_path = f"temp_{int(time.time()*1000)}_{file.filename}"
            cv2.imwrite(temp_path, img)

            cropped_joints, _ = knee_processor.crop_knee_joint_yolo(temp_path)
            if isinstance(cropped_joints, dict) and "error" in cropped_joints:
                errors.append(f"{file.filename}: {cropped_joints['error']}")
                os.remove(temp_path)
                continue
            if len(cropped_joints) == 0:
                errors.append(f"No knee joint detected in {file.filename}")
                os.remove(temp_path)
                continue

            image_sections = []


            general_steps = []
            for title, images_list in {
                'Original': knee_processor.recent_originals,
                'Image Enhancement': knee_processor.recent_clahe,
                'Joints Detected': knee_processor.recent_yolo_annotated
            }.items():
                if len(images_list) > 0:
                    _, buffer = cv2.imencode('.png', images_list[0])
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    general_steps.append({
                        'title': title,
                        'image': f"data:image/png;base64,{img_str}"
                    })

            image_sections.append({
                'section_type': 'general',
                'steps': general_steps
            })

            # Cropped + Resized + Prediction + Grad-CAM + Debug info
            for knee_idx, resized_img in enumerate(knee_processor.recent_resized):
                knee_steps = []

                # Cropped
                if knee_idx < len(knee_processor.recent_cropped):
                    _, buffer = cv2.imencode('.png', knee_processor.recent_cropped[knee_idx])
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    knee_steps.append({
                        'title': 'Cropped',
                        'image': f"data:image/png;base64,{img_str}"
                    })

                # Resized
                _, buffer = cv2.imencode('.png', resized_img)
                img_str = base64.b64encode(buffer).decode('utf-8')
                knee_steps.append({
                    'title': 'Resized',
                    'image': f"data:image/png;base64,{img_str}"
                })

                # Prediction + Grad-CAM + Debug
                try:
                    pred_array, grad_cam_img, debug_info = pipeline2.predict_knee_image(resized_img, return_full=True)


                    prediction_list = []
                    class_labels = {
                        0: "Healthy knee",
                        1: "Mild Osteoarthritis",
                        2: "Moderate Osteoarthritis",
                        3: "Severe Osteoarthritis"
                    }
                    for cls_idx, prob in enumerate(pred_array[0]):
                        prediction_list.append({
                            'label': class_labels[cls_idx],
                            'confidence': f"{prob * 100:.2f}%"
                        })


                    grad_cam_b64 = None
                    if isinstance(grad_cam_img, str):
                        # رجع Base64 جاهز
                        grad_cam_b64 = grad_cam_img
                        debug_info.append(f"Grad-CAM received as Base64 for knee {knee_idx + 1}")
                    elif isinstance(grad_cam_img, np.ndarray) and grad_cam_img.size > 0:
                        # رجع NumPy array → نحوله Base64
                        grad_cam_b64 = pipeline2._gradcam_to_base64(grad_cam_img)
                        debug_info.append(f"Grad-CAM shape: {grad_cam_img.shape}, dtype: {grad_cam_img.dtype}")
                    else:
                        debug_info.append(f"Grad-CAM is empty for knee {knee_idx + 1}")

                    # --- إرسال النتائج للفرونت ---
                    image_sections.append({
                        'section_type': 'knee',
                        'knee_position': f"Knee {knee_idx + 1}",
                        'preprocessing': knee_steps,
                        'result': prediction_list,
                        'explain': grad_cam_b64,
                        'debug_info': debug_info
                    })

                except Exception as e:
                    err_msg = f"Pipeline2 failed for {file.filename} knee {knee_idx + 1}: {str(e)}"
                    errors.append(err_msg)
                    image_sections.append({
                        'section_type': 'knee',
                        'knee_position': f"Knee {knee_idx + 1}",
                        'preprocessing': knee_steps,
                        'result': [],
                        'explain': None,
                        'debug_info': [err_msg]
                    })

            final_output.append({
                'file_name': file.filename,
                'image_sections': image_sections
            })

            os.remove(temp_path)

        except Exception as e:
            err_msg = f"Unexpected error for {file.filename}: {str(e)}"
            errors.append(err_msg)
            traceback.print_exc()

    return jsonify({
        'analysis': final_output,
        'errors': errors
    })



@app.route('/process_all_frames', methods=['POST'])
def process_all_frames():
    data = request.get_json()
    result = {}

    # -------------------------------
    # Frame 1: BMD rules
    # -------------------------------
    if data.get('frame1'):
        f1 = data['frame1']
        try:
            bmd_result = bmd_diagnosis(
                age=f1['age'],
                gender=f1['gender'],
                score_type=f1['score_type'],
                score=f1['score']
            )
            result['frame1'] = {"available": True, "result": bmd_result}
        except Exception as e:
            result['frame1'] = {"available": False, "error": str(e)}

    else:
        result['frame1'] = {"available": False}

    # -------------------------------
    # Frame 2: X-ray images
    # -------------------------------
    if data.get('frame2'):
        frame2_results = []
        files_data = data['frame2']
        for file_data in files_data:
            try:

                img_bytes = base64.b64decode(file_data.split(",")[-1])
                img_np = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

                if img is None:
                    continue

                pipeline_result = pipeline.process(img)


                for knee in pipeline_result.get('knees', []):
                    knee['cropped'] = encode_image_to_base64(knee.get('cropped'))
                    knee['resized'] = encode_image_to_base64(knee.get('resized'))
                    knee['gradcam'] = encode_image_to_base64(knee.get('gradcam'))

                frame2_results.append(pipeline_result)
            except Exception as e:
                frame2_results.append({"error": str(e)})

        result['frame2'] = {"available": True, "result": frame2_results}
    else:
        result['frame2'] = {"available": False}

    # -------------------------------
    # Frame 3: Risk factors model
    # -------------------------------
    if data.get('frame3'):
        f3 = data['frame3']
        try:
            # تحويل بيانات الفريم إلى Features
            features = []
            features.append(int(f3["age"]))
            mapping = {
                "gender": "field_0",
                "hormonal": "field_1",
                "genetic": "field_2",
                "race": "field_3",
                "weight": "field_4",
                "calcium": "field_5",
                "vitamin": "field_6",
                "activity": "field_7",
                "smoking": "field_8",
                "alcohol": "field_9",
                "conditions": "field_10",
                "medications": "field_11",
                "fractures": "field_12",
            }
            for key, field in mapping.items():
                features.append(int(encoders[field].transform([f3[key]])[0]))

            X = np.array([features])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            confidence = round(100 * max(proba), 2)

            result['frame3'] = {
                "available": True,
                "result": {"label": pred, "confidence": confidence}
            }
        except Exception as e:
            result['frame3'] = {"available": False, "error": str(e)}
    else:
        result['frame3'] = {"available": False}

    return jsonify(result)






@app.route('/process_porosis', methods=['POST'])
def process_porosis():
    import traceback
    files = request.files.getlist('xray')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    analysis_results = []
    errors = []

    for file in files:
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            print(f"[DEBUG] Loaded image {file.filename}, shape={img.shape if img is not None else 'None'}")

            if img is None:
                errors.append(f"{file.filename}: Failed to read image")
                continue

            # --- Process with pipeline ---
            result = pipeline.process(img)
            print(f"[DEBUG] Pipeline processed {file.filename}, knees found={len(result.get('knees', []))}")

            # --- Prepare image sections for frontend ---
            image_sections = []


            if 'original' in result:
                image_sections.append({
                    'section_type': 'general',
                    'title': 'Original',
                    'image': encode_image_to_base64(result['original'])
                })


            if 'clahe' in result:
                image_sections.append({
                    'section_type': 'general',
                    'title': 'Enhanced',
                    'image': encode_image_to_base64(result['clahe'])
                })


            if 'yolo_annotated' in result:
                image_sections.append({
                    'section_type': 'general',
                    'title': 'YOLO Annotated',
                    'image': encode_image_to_base64(result['yolo_annotated'])
                })


            for knee in result.get('knees', []):
                image_sections.append({
                    'section_type': 'knee',
                    'knee_position': knee.get('knee_position'),
                    'result': [{
                        'label': knee.get('prediction'),
                        'confidence_percent': knee.get('confidence_percent'),
                        'class_probabilities_percent': knee.get('class_probabilities_percent')
                    }],
                    'images': {
                        'cropped': encode_image_to_base64(knee.get('cropped')),
                        'resized': encode_image_to_base64(knee.get('resized')),
                        'gradcam': encode_image_to_base64(knee.get('gradcam'))
                    }
                })

            analysis_results.append({
                'file_name': file.filename,
                'image_sections': image_sections,
                'warnings': result.get('warnings', [])
            })

        except Exception as e:
            err_msg = f"{file.filename}: Unexpected error: {str(e)}"
            print(err_msg)
            traceback.print_exc()
            errors.append(err_msg)

    return jsonify({'analysis': analysis_results, 'errors': errors})






@app.route('/process_porosis_frame2', methods=['POST'])
def process_porosis_frame2():
    files = request.files.getlist('xray')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    analysis_results = []
    errors = []

    for file in files:
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img is None:
                errors.append(f"{file.filename}: Failed to read image")
                continue


            result = pipeline.process(img)


            image_sections = []


            if 'original' in result:
                image_sections.append({
                    'section_type': 'general',
                    'title': 'Original',
                    'image': encode_image_to_base64(result['original'])
                })


            if 'clahe' in result:
                image_sections.append({
                    'section_type': 'general',
                    'title': 'Enhanced',
                    'image': encode_image_to_base64(result['clahe'])
                })


            if 'yolo_annotated' in result:
                image_sections.append({
                    'section_type': 'general',
                    'title': 'YOLO Annotated',
                    'image': encode_image_to_base64(result['yolo_annotated'])
                })

            # Each knee
            for knee in result.get('knees', []):
                image_sections.append({
                    'section_type': 'knee',
                    'knee_position': knee.get('knee_position'),
                    'result': [{
                        'label': knee.get('prediction'),
                        'confidence_percent': knee.get('confidence_percent'),
                        'class_probabilities_percent': knee.get('class_probabilities_percent')
                    }],
                    'images': {
                        'cropped': encode_image_to_base64(knee.get('cropped')),
                        'resized': encode_image_to_base64(knee.get('resized')),
                        'gradcam': encode_image_to_base64(knee.get('gradcam'))
                    }
                })

            analysis_results.append({
                'file_name': file.filename,
                'image_sections': image_sections,
                'warnings': result.get('warnings', [])
            })

        except Exception as e:
            err_msg = f"{file.filename}: Unexpected error: {str(e)}"
            print(err_msg)
            errors.append(err_msg)

    return jsonify({'analysis': analysis_results, 'errors': errors})




encoders = {
    "field_0": LabelEncoder().fit(["Male", "Female"]),
    "field_1": LabelEncoder().fit(["Normal", "Postmenopausal"]),
    "field_2": LabelEncoder().fit(["Yes", "No"]),
    "field_3": LabelEncoder().fit(["Asian", "Caucasian", "African American"]),
    "field_4": LabelEncoder().fit(["Underweight", "Normal"]),
    "field_5": LabelEncoder().fit(["Low", "Adequate"]),
    "field_6": LabelEncoder().fit(["Sufficient", "Insufficient"]),
    "field_7": LabelEncoder().fit(["Active", "Sedentary"]),
    "field_8": LabelEncoder().fit(["Yes", "No"]),
    "field_9": LabelEncoder().fit(["None", "Moderate"]),
    "field_10": LabelEncoder().fit(["None", "Hyperthyroidism", "Rheumatoid Arthritis"]),
    "field_11": LabelEncoder().fit(["Corticosteroids", "None"]),
    "field_12": LabelEncoder().fit(["Yes", "No"]),
}


@app.route("/process_risk_factors", methods=["POST"])
def process_risk_factors():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    try:
        features = []

        # العمر (من "age" بدل riskAge)
        features.append(int(data["age"]))

        # خريطة بين أسماء الـ JSON وبين الـ encoders
        mapping = {
            "gender": "field_0",
            "hormonal": "field_1",
            "genetic": "field_2",
            "race": "field_3",
            "weight": "field_4",
            "calcium": "field_5",
            "vitamin": "field_6",
            "activity": "field_7",
            "smoking": "field_8",
            "alcohol": "field_9",
            "conditions": "field_10",
            "medications": "field_11",
            "fractures": "field_12",
        }

        for key, field in mapping.items():
            if key not in data:
                return jsonify({"error": f"Missing field {key}"}), 400
            features.append(int(encoders[field].transform([data[key]])[0]))

        X = np.array([features])


        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        confidence = round(100 * max(proba), 2)
        label = pred

        return jsonify({
            "result": {"label": label, "confidence": confidence}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ai_models')
def ai_models():
    return render_template('ai_models.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
