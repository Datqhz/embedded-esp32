import os
from multiprocessing import Value
from PIL import Image
from flask import Flask, request, send_file, jsonify, after_this_request
# from flask_cors import CORS
import numpy as np
import cv2
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.debug = True
# CORS(app)

counter = Value('i', 0)
def save_img(img):
    with counter.get_lock():
        counter.value += 1
        count = counter.value
    img_dir = "esp_image"
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    cv2.imwrite(os.path.join(img_dir, "img_" + str(count) + ".jpg"), img)

@app.route('/receive-image', methods=['POST', 'GET'])
def upload():
    received = request
    img = None
    if received.files:
        print(received.files['imageFile'])
        # convert string of image data to uint8
        file = received.files['imageFile']
        nparr = np.fromstring(file.read(), np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        save_img(img)

        return "[SUCCESS] Image Received", 201
    else:
        return "[FAILED] Image Not Received", 204

@app.route('/process-image', methods=['POST', 'GET'])
def process_img():
    received = request
    img = None
    if received.files:
        print(received.files['imageFile'])
        # convert string of image data to uint8
        file = received.files['imageFile']
        nparr = np.fromstring(file.read(), np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_dir = "esp_image"
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        cv2.imwrite(os.path.join(img_dir, "img_background.jpg"), img)
        facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.imwrite('esp_image/processed_image.jpg', cv2.resize(gray[y:y + h, x:x + w], (12, 12)))
        return "[SUCCESS] Image Process", 201
    else:
        return "[FAILED] Image Not Received To Process", 204

@app.route('/image_2d', methods=['GET'])
def get_image2D_data():
    image_path = "esp_image/processed_image.jpg"

    if not os.path.exists(image_path):
        return "Image not found", 404

    try:
        image = Image.open("esp_image/processed_image.jpg")
        image_array = np.array(image.convert('L'))  # Chuyển về ảnh grayscale
        # image_reshaped = image_array.reshape((1, 12, 12, 1))
        image_reshaped = image_array/255.0
        image_reshaped = image_reshaped.astype(np.float32)
        # Chuyển mảng NumPy thành danh sách để dễ dàng serialize thành JSON
        image_reshaped = image_reshaped.tolist()
        print(image_reshaped)
    except Exception as e:
        return f"Error processing image: {e}", 500
    # this function will be called after response was sent to client
    @after_this_request
    def remove_file(response):
        try:
            os.remove(image_path)
            app.logger.info(f"Deleted file: {image_path}")
        except Exception as e:
            app.logger.error(f"Error removing file: {e}")
        return response
    # Trả về dữ liệu hình ảnh dưới dạng JSON
    return jsonify(image_reshaped)

@app.route('/save-open-door', methods=['POST'])
def receive_data():
    try:
        # Nhận dữ liệu JSON từ thiết bị
        data = request.get_json()

        # Kiểm tra xem JSON có chứa thuộc tính "idx" không
        if 'idx' in data:
            idx = data['idx']
            # Xử lý dữ liệu ở đây, ví dụ: lưu vào cơ sở dữ liệu, thực hiện tính toán, vv.
            name = ""
            if(idx == 0):
                name = "Đạt"
            elif(idx == 1):
                name = "Long"
            else:
                name = "Quỳnh"

            image_path = "esp_image/img_background.jpg"

            multipart_data = MultipartEncoder(
                fields={
                    "name": name,
                    "image": ('filename', open(image_path, 'rb'), 'image/jpeg,image/png,image/jpg')
                }
            )
            response = requests.post(
                'http://192.168.1.20:3000/api/add-history',
                data=multipart_data,
                headers={'Content-Type': multipart_data.content_type}
            )
            if response.status_code == 200:
                print('Product added successfully:', response.json())
            else:
                print('Error adding product:', response.status_code, response.text)
            # Trả về phản hồi cho thiết bị
            response = {'message': 'Data received successfully', 'idx': idx}
            return jsonify(response), 200
        else:
            # Trả về lỗi nếu không có thuộc tính "idx"
            response = {'error': 'No "idx" attribute found in JSON'}
            return jsonify(response), 400
    except Exception as e:
        # Trả về lỗi nếu có bất kỳ lỗi nào xảy ra trong quá trình xử lý dữ liệu
        response = {'error': str(e)}
        return jsonify(response), 500

if __name__ == '__main__':
    app.run(host='192.168.1.20',port=5000)

