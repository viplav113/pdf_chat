from flask import Flask, request, render_template
import pdfplumber
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        uploaded_file = request.files["pdf"]
        question = request.form["question"]

        if uploaded_file:
            pdf_text = extract_text_from_pdf(uploaded_file)
            answer = get_answer(question, pdf_text)
            return render_template("index.html", answer=answer)
        else:
            return "No PDF file uploaded."

    return render_template("index.html", answer="")

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_answer(question, context):
    return qa_pipeline(question=question, context=context)

if __name__ == "__main__":
    app.run(debug=True)
