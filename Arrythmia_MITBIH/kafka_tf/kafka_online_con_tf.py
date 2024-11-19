import tensorflow as tf
import tensorflow_io as tfio
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# Kafka Configuration
kafka_broker = "localhost:9092"
features_topic = "ha-tf"
num_columns = 187  # Number of feature columns
batch_size = 32


# Define a simple binary classification model
def build_binary_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize the model
model = build_binary_model(input_dim=num_columns)

online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(
    topics=["susy-test"],
    group_id="cgonline",
    servers="localhost:9092",
    stream_timeout=10000,
    configuration=[
        "session.timeout.ms=7000",
        "max.poll.interval.ms=8000",
        "auto.offset.reset=earliest"
    ],
)


def decode_kafka_online_item(raw_message, raw_key):
    """
    Decode Kafka messages into features and labels.
    """
    message = tf.io.decode_csv(raw_message, [[0.0] for _ in range(1, 188)])
    key = tf.strings.to_number(raw_key)
    return (message, key)


counter = 0
result_dict = {0: "Normal", 1: "Arrythmia"}

# Online training
for mini_ds in online_train_ds:
    counter += 1
    mini_ds = mini_ds.shuffle(buffer_size=32)
    mini_ds = mini_ds.map(decode_kafka_online_item)
    mini_ds = mini_ds.batch(32)

    if counter > 2:
        predictions = model.predict(mini_ds, verbose=1)

        # Collect actual and predicted values for metrics calculation
        actuals = []
        predicted_classes = []
        predicted_probs = []

        for features, labels in mini_ds:
            actuals.extend(labels.numpy())
            batch_predicted_probs = model.predict(features, verbose=0)
            predicted_probs.extend(batch_predicted_probs)
            predicted_classes.extend((batch_predicted_probs > 0.5).astype(int))

        # Convert lists to arrays for sklearn metrics
        actuals = tf.convert_to_tensor(actuals)
        predicted_classes = tf.convert_to_tensor(predicted_classes)
        predicted_probs = tf.convert_to_tensor(predicted_probs)

        # Calculate metrics
        f1 = f1_score(actuals, predicted_classes)
        roc_auc = roc_auc_score(actuals, predicted_probs)
        tn, fp, fn, tp = confusion_matrix(actuals, predicted_classes).ravel()
        sensitivity = tp / (tp + fn)  # Recall
        specificity = tn / (tn + fp)

        # Print metrics
        print(f"F1 Score: {f1:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Print actual and predicted values
        for actual, predicted in zip(actuals.numpy(), predicted_classes.numpy()):
            print(f"Actual: {result_dict[actual]}, Predicted: {result_dict[predicted[0]]}")

    # Evaluate the model on the current mini-batch
    val_loss, val_accuracy = model.evaluate(mini_ds, verbose=1)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Fit the model on the current mini-batch
    model.fit(mini_ds, epochs=10)
