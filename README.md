# Osteo-by-AI 🤖🦴🩻
End-to-end AI framework for osteoarthritis and osteoporosis diagnosis

## Abstarct
This project, titled “Osteo by AI”, introduces an intelligent diagnostic system for detecting two major bone disorders: osteoarthritis and osteoporosis.

It employs an integrated AI framework combining deep learning, ensemble modeling, and explainable AI techniques for automated and accurate diagnosis. A YOLO-based helper model is used for knee joint localization, followed by transfer learning for OA classification and a multi-source ensemble approach for OP diagnosis, leveraging imaging, clinical, and rule-based submodels.

Data preprocessing, augmentation, and Grad-CAM visualization enhance both model performance and interpretability. Developed and tested on publicly available datasets, the system achieved promising results and was deployed via a web-based interface for real-time clinical support, with further validation recommended for broader application.

The project aims to contribute to early detection and monitoring of bone diseases, especially in regions with limited access to medical expertise. Results demonstrate promising accuracy, confirming the potential of artificial intelligence in medical imaging and diagnostics.

## 1️⃣ Osteoarthritis (OA)

### 📄 System Design

![OA_System](images/OA_System.png)

### 🪜 Preprocessing Steps

![OA_images_preprocessing](images/OA_images_preprocessing.png)

### 🔎 Results

<img src="images/OA_model_results.png" alt="OA_model_results" width="50%"/>


Accuracy = .93

## 🪞Front-End


  <img src="images/Screenshot 2025-09-29 230347.png" alt="الصورة الأولى" width="50%"/>
  <img src="images/Screenshot 2025-09-29 230736.png" alt="الصورة الثانية" width="50%"/>
  <img src="images/Screenshot 2025-09-29 230910.png" alt="الصورة الثالثة" width="50%"/>





## 2️⃣ Osteoporosis (OP)

### 📄 System Design

![OP_System](images/OP_System.png)

### 🪜 Preprocessing Steps (Images)

![OP_images_Preprocessing](images/OP_images_Preprocessing.jpg)

### 🔎 Results (Images Model)

<img src="images/OP_model1_results.jpg" alt="OP_model1_results" width="50%"/>


Accuracy = .81

### 🔎 Results (Risk Factors Model)

<img src="images/OP_model2_results.jpg" alt="OP_model2_results" width="50%"/>

Accuracy = .91

### 🪞 Front-End


  <img src="images/Screenshot 2025-09-29 231138.png" alt="الصورة الأولى" width="50%"/>
  <img src="images/Screenshot 2025-09-29 231156.png" alt="الصورة الثانية" width="50%"/>
  <img src="images/Screenshot 2025-09-29 231504.png" alt="الصورة الثالثة" width="50%"/>






## ✨Helper Model : YOLOv8 Training — [Roboflow](https://roboflow.com)

![YOLO_Training](images/YOLO_Training.png)




## Authors

👩🏻‍💻 Rama Amjad Alsadeq <br>
👩🏻‍💻 Oula Saleem Hanandeh <br>
👩🏻‍💻 Shaima Feras Alharahsheh <br>

This project, in its first version, was developed as a fulfillment of the graduation requirements for the Data Science and Artificial Intelligence program at Al al-Bayt University, Jordan, in August 2025.

