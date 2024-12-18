from flask import Flask, request, render_template
import joblib
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Muat model dan vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Unduh dan setel stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Fungsi untuk praproses teks
def preprocess_text(text):
    text = text.lower()  # Ubah teks menjadi huruf kecil
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])  # Hapus tanda baca dan angka
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Hapus stop words
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = preprocess_text(message)
        vect = vectorizer.transform([data])
        prediction = model.predict(vect)
        result = 'spam' if prediction[0] == 'spam' else 'ham (non-spam)'
        return render_template('index.html', prediction_text='Pesan ini adalah: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
