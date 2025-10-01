import os
from flask import Flask, request, redirect, render_template, flash, url_for
from werkzeug.utils import secure_filename
import shutil
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np

#==========================================================
# 猫、犬、狐 判別器
#==========================================================
classes = ["猫","犬","狐"]
image_size = 128

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#　学習済みモデルをロード
model = load_model('./catdogfox_model.keras')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # uploadされた画像ファイルを保存
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            # staticフォルダに画像ファイルをコピー
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            static_filepath = os.path.join(STATIC_FOLDER, filename)
            shutil.copy(filepath, static_filepath)

            # 受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, color_mode='rgb', target_size=(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            
            # 変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"
            img_url = url_for(STATIC_FOLDER, filename=filename)
            #img_url = url_for(UPLOAD_FOLDER, filename=filename)

            return render_template("index.html", answer=pred_answer, img_url=img_url)

    return render_template("index.html", answer="", img_url="")


# if __name__ == "__main__":
#     app.run()
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)