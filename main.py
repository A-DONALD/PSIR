import os
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from transformers import TFBertModel, BertTokenizer
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

# Step 1: Load Dataset
def load_dataset(data_dir):
    texts, labels = [], []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")
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
def bert_embedding(texts, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    def batch_tokenize_and_embed(texts_batch):
        inputs = tokenizer(texts_batch, return_tensors='tf', padding=True, truncation=True, max_length=512)
        outputs = bert_model(inputs).last_hidden_state
        cls_embeddings = outputs[:, 0, :]  # Use [CLS] token embeddings
        return cls_embeddings.numpy()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        embeddings.append(batch_tokenize_and_embed(batch_texts))
    return np.vstack(embeddings)

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
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
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
    import os

    # Dynamically set the dataset path based on the script's location
    data_dir = os.path.join(os.getcwd(), 'dataset')
    logging.info("Loading dataset...")

    try:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")
        texts, labels = load_dataset(data_dir)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Encode labels as integers
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(texts, int_labels, test_size=0.2, random_state=42)

    # TF-IDF + MultinomialNB
    logging.info("Training TF-IDF + MultinomialNB...")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_vectorization(X_train, X_test)
    nb_model_tfidf, nb_report_tfidf = train_naive_bayes(X_train_tfidf.toarray(), y_train, X_test_tfidf.toarray(), y_test)
    logging.info(f"TF-IDF + MultinomialNB:\n{nb_report_tfidf}")

    # TF-IDF + TensorFlow
    logging.info("Training TF-IDF + TensorFlow...")
    nn_model_tfidf, nn_report_tfidf = train_tensorflow_nn(X_train_tfidf.toarray(), y_train, X_test_tfidf.toarray(), y_test, len(unique_labels))
    logging.info(f"TF-IDF + TensorFlow:\n{nn_report_tfidf}")

    # BERT + MultinomialNB
    logging.info("Training BERT + MultinomialNB...")
    X_train_bert = bert_embedding(X_train)
    X_test_bert = bert_embedding(X_test)
    nb_model_bert, nb_report_bert = train_naive_bayes(X_train_bert, y_train, X_test_bert, y_test)
    logging.info(f"BERT + MultinomialNB:\n{nb_report_bert}")

    # BERT + TensorFlow
    logging.info("Training BERT + TensorFlow...")
    nn_model_bert, nn_report_bert = train_tensorflow_nn(X_train_bert, y_train, X_test_bert, y_test, len(unique_labels))
    logging.info(f"BERT + TensorFlow:\n{nn_report_bert}")


if __name__ == "__main__":
    main()
