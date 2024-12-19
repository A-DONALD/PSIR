import os
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from transformers import TFBertModel, BertTokenizer

# Step 1: Load Dataset
def load_dataset(data_dir):
    texts, labels = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                    labels.append(label)
    return texts, labels

# Step 2: TF-IDF Vectorization
def tfidf_vectorization(train_texts, test_texts):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)
    return X_train_tfidf, X_test_tfidf, tfidf

# Step 3: BERT Embedding
def bert_embedding(train_texts, test_texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    def tokenize_and_embed(texts):
        inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=512)
        outputs = bert_model(inputs)[0]
        return tf.reduce_mean(outputs, axis=1).numpy()

    X_train_bert = tokenize_and_embed(train_texts)
    X_test_bert = tokenize_and_embed(test_texts)
    return X_train_bert, X_test_bert

# Step 4: Multinomial Naive Bayes Model
def train_naive_bayes(X_train, y_train, X_test, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return model, report

# Step 5: TensorFlow Neural Network Model
def train_tensorflow_nn(X_train, y_train, X_test, y_test, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred)
    return model, report

# Main Function
def main():
    # Load dataset
    data_dir = '/dataset'
    texts, labels = load_dataset(data_dir)

    # Encode labels as integers
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = [label_to_int[label] for label in labels]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(texts, int_labels, test_size=0.2, random_state=42)

    # TF-IDF + MultinomialNB
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_vectorization(X_train, X_test)
    nb_model_tfidf, nb_report_tfidf = train_naive_bayes(X_train_tfidf, y_train, X_test_tfidf, y_test)
    print("TF-IDF + MultinomialNB:\n", nb_report_tfidf)

    # TF-IDF + TensorFlow
    nn_model_tfidf, nn_report_tfidf = train_tensorflow_nn(X_train_tfidf, np.array(y_train), X_test_tfidf, np.array(y_test), len(unique_labels))
    print("TF-IDF + TensorFlow:\n", nn_report_tfidf)

    # BERT + MultinomialNB
    X_train_bert, X_test_bert = bert_embedding(X_train, X_test)
    nb_model_bert, nb_report_bert = train_naive_bayes(X_train_bert, y_train, X_test_bert, y_test)
    print("BERT + MultinomialNB:\n", nb_report_bert)

    # BERT + TensorFlow
    nn_model_bert, nn_report_bert = train_tensorflow_nn(X_train_bert, np.array(y_train), X_test_bert, np.array(y_test), len(unique_labels))
    print("BERT + TensorFlow:\n", nn_report_bert)

if __name__ == "__main__":
    main()