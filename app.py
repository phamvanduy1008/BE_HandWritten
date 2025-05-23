import os
import numpy as np
import cv2
from PIL import Image
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from svm_model import SVM, knn_predict
import pandas as pd
import urllib.request

app = Flask(__name__)
CORS(app)


# Link tải mnist_train.csv từ Google Drive (dạng direct)
train_path = "./data/mnist_train.csv"
train_url = "https://drive.google.com/uc?export=download&id=17ga2u6S0D0KUewTs9XC69yKR61hljCDq"

# Tải file nếu chưa có
if not os.path.exists(train_path):
    print("Downloading mnist_train.csv from Google Drive...")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    urllib.request.urlretrieve(train_url, train_path)
    print("Download complete.")

# Load dữ liệu huấn luyện
train = pd.read_csv(train_path)
X_train = train.iloc[:, 1:].values.astype(np.float32) / 255.0
y_train = train.iloc[:, 0].values


try:
    loaded_model = joblib.load("./model/svm_model.pkl")
    print("Mô hình đã được tải thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    loaded_model = None

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return jsonify({"error": "Mô hình chưa được tải"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Không có file nào được chọn"}), 400

    try:
        # Đọc và tiền xử lý ảnh nâng cao
        img = Image.open(file.stream).convert('L')  # Chuyển thành grayscale
        img_resized = img.resize((28, 28), Image.BILINEAR)
        img_array = np.array(img_resized).astype(np.uint8)  # Chuyển sang uint8 để xử lý OpenCV

        # Làm sạch ảnh: làm nét vẽ rõ hơn, loại bỏ nhiễu
        _, img_binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)  # Chuyển thành ảnh nhị phân
        kernel = np.ones((3, 3), np.uint8)
        img_dilated = cv2.dilate(img_binary, kernel, iterations=1)
        
        img_array = img_dilated.astype(np.float32) / 255.0  # Chuẩn hóa lại về [0, 1]
        img_array = img_array.reshape(1, 784)  # Reshape thành vector 1x784

        prediction = loaded_model.predict(img_array, 0, 1, X_train, y_train)[0]
        svm_scores = np.dot(loaded_model.W, img_array.T) + loaded_model.b[:, np.newaxis]
        svm_prediction = np.argmax(svm_scores, axis=0)[0]

        if loaded_model.num_uncertain == 1:
            print("Số ảnh không chắc chắn: 1")
            print(f"SVM: {svm_prediction}")
            print(f"KNN: {prediction}")
            print(f"Kết quả: {prediction}")
        else:
            print(f"SVM: {svm_prediction}")
            print(f"Kết quả: {prediction}")

        return jsonify({
            "prediction": int(prediction),
            "svm_prediction": int(svm_prediction),
            "uncertain": loaded_model.num_uncertain == 1
        })

    except Exception as e:
        return jsonify({"error": f"Lỗi khi xử lý ảnh: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)