import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
df = pd.read_csv("spam.csv", encoding='latin-1')

# Drop unnecessary columns (Kaggle dataset has extra columns like 'Unnamed: 2', etc.)
df = df[['v1', 'v2']]  # v1 = label, v2 = text

# Rename columns
df.columns = ['label', 'text']

# Remove empty messages
df.dropna(subset=['text'], inplace=True)

# Convert labels to binary: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ------------------------------
# 2️⃣ Split features and labels
# ------------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 3️⃣ TF-IDF Vectorization
# ------------------------------
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------------------
# 4️⃣ Train Multinomial Naive Bayes
# ------------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Test accuracy
accuracy = model.score(X_test_tfidf, y_test)
print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# ------------------------------
# 5️⃣ Save model and vectorizer
# ------------------------------
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("✅ Training completed successfully! Model and vectorizer saved as .joblib")
