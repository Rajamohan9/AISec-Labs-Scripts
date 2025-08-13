####### Create Directories

rm -rf sequential-ai-model
mkdir sequential-ai-model
cd sequential-ai-model

####### Install Requirements
cat>requirements.txt<<EOF
tensorflow==2.11.0
scikit-learn==1.2.0
numpy==1.23.5
EOF

pip install -r requirements.txt

####### building-sequential-model.py

cat>building-sequential-model.py<<EOF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib

# Training data
data = [
    ("What is AI?", 0),
    ("Define ML.", 1),
    ("Explain deep learning.", 2),
    ("What is Python?", 3),
    ("Define CPU.", 4),
    ("Define GPU.", 5),
    ("What is NLP?", 6),
    ("What is data science?", 7),
    ("What is optimizer?", 8),
    ("What is gradient descent?", 9)
]

responses = [
    "AI stands for Artificial Intelligence.",
    "Machine Learning is a branch of AI.",
    "Deep learning uses neural networks.",
    "Python is a popular programming language.",
    "CPU stands for Central Processing Unit.",
    "GPU is Graphics Processing Unit.",
    "Natural Language Processing is a part of AI.",
    "It is a field of analyzing data.",
    "Algorithm to minimize loss.",
    "Optimization method."
]

# Prepare training data
X_train = [item[0] for item in data]
y_train = [item[1] for item in data]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train).toarray()
y_vec = to_categorical(y_train, num_classes=10)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_vec.shape[1],)),
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_vec, y_vec, epochs=30, verbose=0)

# Save model
model.save("sequential_model.h5")

# Save vectorizer and responses
joblib.dump(vectorizer, "vectorizer.joblib")

with open("responses.json", "w") as f:
    json.dump(responses, f)

print("[✔] Sequential model trained and saved as sequential_model.h5")
EOF


cat>sequential-model-consumer.py<<EOF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import json
import joblib
from tensorflow.keras.models import load_model

# Load model and assets
model = load_model("sequential_model.h5")
vectorizer = joblib.load("vectorizer.joblib")

with open("responses.json", "r") as f:
    responses = json.load(f)

# Prompt and inference
prompt = input("Ask a question: ")
X_input = vectorizer.transform([prompt]).toarray()
prediction = model.predict(X_input)
predicted_label = np.argmax(prediction)

print("Response:", responses[predicted_label])
EOF

python3 building-sequential-model.py
