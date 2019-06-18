from flask import Flask
from keras.models import *
from keras.layers import *
import sys
import os
from PIL import Image

from flask import jsonify
from flask import Flask, flash, request, redirect, url_for, render_template

app = Flask(__name__)

sys.path.append(os.path.dirname(__file__))


def predict(img):
    os.chdir(os.path.dirname(__file__))
    import numpy as np
    import string
    characters = string.digits + string.ascii_uppercase
    width, height, n_len, n_class = 170, 80, 4, len(characters) + 1
    rnn_size = 128

    input_tensor = Input((width, height, 3))
    x = input_tensor
    for i in range(3):
        x = Convolution2D(32, 3, 3, activation='relu')(x)
        x = Convolution2D(32, 3, 3, activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    x = Dense(32, activation='relu')(x)
    gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
    gru1_merged = merge([gru_1, gru_1b], mode='sum')
    gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
    x = merge([gru_2, gru_2b], mode='concat')
    x = Dropout(0.25)(x)
    x = Dense(n_class, init='he_normal', activation='softmax')(x)
    base_model = Model(input=input_tensor, output=x)
    base_model.load_weights(os.path.join(app.root_path, 'model.h5'))

    X = np.zeros((1, width, height, 3), dtype=np.uint8)
    X[0] = np.array(img).transpose(1, 0, 2)
    y_pred = base_model.predict(X)
    y_pred = y_pred[:, 2:, :]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
    out = ''.join([characters[x] for x in out[0]])
    return str(out)


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "file must in files"})
        file = request.files['file']
        img_path = os.path.join(app.static_folder, file.filename)
        file.save(img_path)
        from PIL import Image
        img = Image.open(img_path)
        return predict(img)


@app.route('/demo', methods=['GET', 'POST'])
def demo(name=None):
    if request.method == "POST":
        from captcha.image import ImageCaptcha
        import string
        characters = string.digits + string.ascii_uppercase
        width, height, n_len, n_class = 170, 80, 4, len(characters) + 1
        user_inputs = request.get_json()
        if user_inputs.get("origin_string", None):
            image_name = request.get_json()["origin_string"]
            ImageCaptcha(width=width, height=height)
            img = ImageCaptcha(width=width, height=height).generate_image(image_name)
            img_file = "{}.png".format(image_name)
            img.save(os.path.join(app.static_folder, img_file))
            return jsonify({
                "url": "/static/{}".format(img_file)
            })
        if user_inputs.get("filename", None):
            filename = user_inputs.get("filename", None)
            img_path = os.path.join(app.static_folder, filename)
            img = Image.open(img_path)
            return predict(img)

    return render_template('demo.html', name=name)


@app.route('/')
def hello_world(name=None):
    return render_template('index.html', name=name)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
