from kafka import KafkaConsumer
import json
from river import metrics, compose, preprocessing
from deep_river.classification import Classifier
from torch import nn
import matplotlib.pyplot as plt
import time

# Kafka configuration
bootstrap_servers = "localhost:9092"
topic_name = "Arr-river"  # Kafka topic for the training data

# Initialize metrics for tracking model performance
accuracy = metrics.Accuracy()
f1_score = metrics.MacroF1()  # Macro F1 for multi-class tasks
sensitivity_metric = metrics.Recall()

# Initialize a dictionary of ROC AUC metrics, one per class
roc_auc_dict = {i: metrics.ROCAUC() for i in range(5)}  # Assuming 5 classes


# Define a custom PyTorch model for multi-class classification
class MyDeepModel(nn.Module):
    def __init__(self, n_features, n_classes=5):  # 5 classes
        super(MyDeepModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),  # Output layer for 5 classes
        )

    def forward(self, x):
        return self.model(x)


# Initialize the deep learning-based classifier
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    Classifier(
        module=MyDeepModel,
        loss_fn="cross_entropy",
        optimizer_fn="adam",
        lr=0.001,
        output_is_logit=True  # Cross-entropy expects raw logits
    )
)

# Lists to store metrics for plotting
accuracies, f1_scores, sensitivities, roc_aucs = [], [], [], []

# Message counter
count = 0

# Track the time of the last message
last_message_time = time.time()
timeout = 120  # 2-minute timeout

# Initialize the Kafka consumer
consumer = KafkaConsumer(
    topic_name,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset="earliest",
    group_id="river-consumer-group",
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    key_deserializer=lambda x: x.decode('utf-8')
)


# Function to process each Kafka message
def process_message(key, value):
    global count
    y_true = int(float(key))  # Convert key (output label) to integer

    # Convert all feature values to floats
    features = {k: float(v) for k, v in value.items()}

    # Make a prediction and get probabilities
    y_pred_proba = model.predict_proba_one(features)
    y_pred = max(y_pred_proba, key=y_pred_proba.get) if y_pred_proba else None

    if y_pred is not None:
        accuracy.update(y_true, y_pred)
        f1_score.update(y_true, y_pred)
        sensitivity_metric.update(y_true, y_pred)

        # Update ROC AUC for each class using one-vs-rest approach
        for class_label, roc_auc_metric in roc_auc_dict.items():
            # Treat the current class as positive and all others as negative
            y_true_binary = 1 if y_true == class_label else 0
            y_pred_proba_class = y_pred_proba.get(class_label, 0.0)
            roc_auc_metric.update(y_true_binary, y_pred_proba_class)

    # Store metrics every 3000 messages or final batch
    count += 1
    if count % 10000 == 0 or count > 120000:
        print(f"Processed {count} messages")
        accuracies.append(accuracy.get() * 100)
        f1_scores.append(f1_score.get() * 100)
        sensitivities.append(sensitivity_metric.get() * 100)

        # Calculate average ROC AUC across classes
        average_roc_auc = sum(roc_auc.get() for roc_auc in roc_auc_dict.values()) / len(roc_auc_dict)
        roc_aucs.append(average_roc_auc * 100)

        print(f"Accuracy: {accuracies[-1]:.2f}%")
        print(f"F1 Score: {f1_scores[-1]:.2f}%")
        print(f"Sensitivity: {sensitivities[-1]:.2f}%")
        print(f"Average ROC AUC: {roc_aucs[-1]:.2f}%")

    model.learn_one(features, y_true)


# Function to plot metrics
def plot_metrics():
    plt.figure(figsize=(10, 6))

    # Plot each metric over time
    plt.plot(accuracies, label="Accuracy")
    plt.plot(f1_scores, label="F1 Score")
    plt.plot(sensitivities, label="Sensitivity")
    plt.plot(roc_aucs, label="Average ROC AUC")

    # Labeling
    plt.xlabel("Batches of 3000 Messages")
    plt.ylabel("Percentage")
    plt.title("Model Performance Metrics Over Time")
    plt.legend()
    plt.show()


# Consume and process messages from Kafka
try:
    print("Starting Kafka Consumer...")
    while True:
        msg_pack = consumer.poll(timeout_ms=1000)

        if msg_pack:
            for tp, messages in msg_pack.items():
                for message in messages:
                    last_message_time = time.time()
                    key = message.key
                    value = message.value
                    process_message(key, value)

        if time.time() - last_message_time >= timeout:
            print("No messages received for 2 minutes. Plotting final metrics and closing consumer.")
            plot_metrics()
            break

except KeyboardInterrupt:
    print("Consumer interrupted manually. Closing...")

except Exception as e:
    print(f"Error processing message: {e}")

finally:
    consumer.close()
