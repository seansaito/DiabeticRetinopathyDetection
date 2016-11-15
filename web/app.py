import base64
from StringIO import StringIO
from PIL import Image
from flask import Flask, request, render_template
from retinopathy import max_convert, compute_score


app = Flask(__name__)


def encode_image(pil_img):
    img_io = StringIO()
    pil_img.convert('RGB').save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return base64.b64encode(img_io.buf)


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/detection', methods=['POST'])
def detection():
    file = request.files['image']
    img = Image.open(file)
    cropped = max_convert(img, 512, stretch=False)
    score = compute_score(cropped)
    score_words = {
        0: "0 - No Diabetic Retinopathy",
        1: "1 - Mild Diabetic Retinopathy",
        2: "2 - Moderate Diabetic Retinopathy",
        3: "3 - Severe Diabetic Retinopathy",
        4: "4 - Proliferative Diabetic Retinopathy",
    }
    return render_template('detection.html', img_data=encode_image(cropped), score=score_words[score])
    # return render_template('homepage.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run()
