import tkinter as tk
from tkinter import messagebox
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download stopwords once
nltk.download('stopwords')

# Load dataset (use the SMS Spam Collection dataset)
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing
ps = PorterStemmer()
corpus = []

for msg in df['message']:
    review = re.sub('[^a-zA-Z]', ' ', msg)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))

# Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(corpus).toarray()
y = df['label'].values

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# GUI function
def classify_message():
    msg = input_box.get("1.0", "end").strip()
    if not msg:
        messagebox.showwarning("Input Required", "Please enter a message to classify.")
        return
    review = re.sub('[^a-zA-Z]', ' ', msg).lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    final_input = vectorizer.transform([' '.join(review)]).toarray()
    pred = model.predict(final_input)[0]
    result = "🚫 SPAM" if pred == 1 else "✅ NOT SPAM"
    result_label.config(text=result, fg="red" if pred == 1 else "green")

# GUI setup
root = tk.Tk()
root.title("Spam Mail Classifier")
root.geometry("500x400")
root.configure(bg="white")

tk.Label(root, text="📧 Spam Mail Classifier", font=("Arial", 18, "bold"), bg="white").pack(pady=10)
input_box = tk.Text(root, height=8, width=55, font=("Arial", 12))
input_box.pack(pady=10)
tk.Button(root, text="Check Spam", font=("Arial", 14), command=classify_message).pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 16), bg="white")
result_label.pack(pady=10)

root.mainloop()