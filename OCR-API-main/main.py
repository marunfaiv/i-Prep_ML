import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
from imutils.contours import sort_contours
import imutils
import PyPDF2
from io import BytesIO
import spacy
from sentence_transformers import SentenceTransformer
from waitress import serve

app = Flask(__name__)
model = load_model("./model/handwriting.h5")
resume_ner_model = spacy.load("./model/ner-model")
embedding_model = SentenceTransformer("./model/all-MiniLM-L6-v2")


def api_error(code, message, http_code):
    return jsonify({"code": code, "message": message}), http_code


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    chars = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            roi = gray[y : y + h, x : x + w]
            thresh = cv2.threshold(
                roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            (tH, tW) = thresh.shape
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)
            padded = cv2.copyMakeBorder(
                thresh,
                top=dY,
                bottom=dY,
                left=dX,
                right=dX,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            padded = cv2.resize(padded, (32, 32))
            padded = padded.astype("float32")
            padded = np.expand_dims(padded, axis=-1)
            chars.append((padded, (x, y, w, h)))

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")
    preds = model.predict(chars)

    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames += "abcdefghijklmnopqrstuvwxyz"
    labelNames = [l for l in labelNames]
    output = ""
    for pred, (x, y, w, h) in zip(preds, boxes):
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        output += label

    return output


def readPDF(file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return api_error("FILE_REQUIRED", "File field is required", 400)

    file = request.files["file"]
    file_mime_type = request.form["filetype"]    

    if file_mime_type == "image/jpeg" or file_mime_type == "image/png":
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        content = process_image(image)
    elif file_mime_type == "application/pdf":
        pdfContent = readPDF(file)

        if pdfContent == "":
            return api_error("INVALID_FILE", "Invalid file", 400)
        else:
            content = pdfContent
    else:
        return api_error("INVALID_FILE", "Invalid file", 400)

    predicted = resume_ner_model(content)

    skills = []
    for ent in predicted.ents:
        if ent.label_.upper() == "SKILLS":
            vector = embedding_model.encode(ent.text)

            skills.append(
                {
                    "skill": ent.text,
                    "vector": vector.tolist(),
                }
            )

    return jsonify({"result": [skill for skill in skills]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
