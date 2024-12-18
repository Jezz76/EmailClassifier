import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.stem import WordNetLemmatizer
import string
import joblib

# Unduh stop words dan lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# Fungsi untuk praproses teks
def preprocess_text(text):
    text = text.lower()  # Ubah teks menjadi huruf kecil
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])  # Hapus tanda baca dan angka
    words = text.split()
    # Lemmatization dan menghilangkan stop words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Muat dataset
data = pd.read_csv('archive/spam.csv', encoding='latin-1', header=None, usecols=[0, 1], skiprows=1)

# Setel nama kolom agar lebih mudah digunakan
data.columns = ['label', 'text']

# Terapkan fungsi praproses pada kolom teks
data['text'] = data['text'].apply(preprocess_text)

# Bagi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=1)

# Vectorize teks dengan n-grams (bigrams)
vectorizer = CountVectorizer(ngram_range=(1, 2))  # Ini menghasilkan unigram dan bigram
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Latih model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluasi model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Fungsi untuk uji spam dengan input baru
def predict_spam(text):
    text = preprocess_text(text)
    text_vector = vectorizer.transform([text])
    return model.predict(text_vector)

# Uji pesan contoh
print(predict_spam("Congratulations! You've won a free ticket to Bahamas."))

# Simpan model dan vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
