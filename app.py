import os
import cv2
import time
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('select.html')

def load(filename):
    img = image.load_img(filename, target_size=(128, 128))
    np_image = image.img_to_array(img)
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

@app.route('/predict', methods=['POST']) ## Fungsi predict untuk prediksi file yang diinputkan
def predict():
    chosen_model = request.form['select_model']
    model = load_model('mod6/mod5.h5') ## Hasil model training yang disimpan pada effnetmod5.h5
    file = request.files["file"]
    file.save(os.path.join('mod6/static','temp.jpg')) #untuk menyimpan file yang diunggah ke sistem file
    img = load('mod6/static/temp.jpg') #Fungsi ini membaca gambar dari file "temp.jpg" dan mengonversinya menjadi format yang dapat digunakan oleh model
    start = time.time() #Merekam waktu awal sebelum prediksi dimulai
    pred = model.predict(img) #Memprediksi kelas gambar yang telah dimuat menggunakan model yang telah dimuat sebelumnya Hasil prediksi disimpan dalam variabel pred
    labels = np.argmax(pred, axis=-1) #Mengambil indeks kelas dengan probabilitas tertinggi dari hasil prediks
    print(labels) #Mencetak indeks kelas yang diprediksi ke konsol
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred.flatten()] #Mengonversi probabilitas prediksi menjadi persentase dan menyimpannya dalam bentuk list
    return predict_result(chosen_model, runtimes, respon_model, 'temp.jpg') #Mengembalikan hasil prediksi dengan memanggil fungsi predict_result dengan parameter yang sesuai

def predict_result(model, run_time, probs, img): # untuk menyusun hasil prediksi dalam bentuk yang dapat ditampilkan pada halaman web menggunakan Flask
     # Mapping indeks kelas ke nama kelas
    class_list = {'Rock': 0, 'Paper': 1, 'Scissor':2}
    # Menentukan indeks kelas yang diprediksi dengan probabilitas tertinggi
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys()) # Mendapatkan daftar nama kelas
     # Merender template HTML dengan hasil prediksi
    return render_template('/result_select.html', labels=labels,
                            probs=probs, model=model, pred=idx_pred,
                            run_time=run_time, img=img)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=2000)