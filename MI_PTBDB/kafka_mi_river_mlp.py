from kafka import KafkaConsumer
import json
from river import metrics
from river import linear_model, preprocessing
from river import forest
from river import ensemble, tree, preprocessing
import matplotlib.pyplot as plt
import time
from river import ensemble
from river import metrics, datasets, compose, preprocessing
from deep_river.classification import Classifier
from torch import nn
from tqdm import tqdm

# Kafka configuration
bootstrap_servers = "localhost:9092"  # Update with your Kafka server address
topic_name = "MI-river"  # Kafka topic for the training data

# Initialize metrics for tracking model performance
accuracy = metrics.Accuracy()
f1_score = metrics.F1()
roc_auc = metrics.ROCAUC()
sensitivity_metric = metrics.Recall()  # Sensitivity is the same as Recall
# Specificity will be calculated manually

# Initialize a simple River model for demonstration

# Define a custom PyTorch model
class MyDeepModel(nn.Module):
    def __init__(self, n_features):
        super(MyDeepModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Adjust output size for number of classes
            nn.Sigmoid()
            #nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the deep learning-based classifier
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    Classifier(
        module=MyDeepModel,
        loss_fn="binary_cross_entropy",
        optimizer_fn="adam",
        lr=0.001,
        output_is_logit=False
    )
)

# Lists to store metrics for plotting
accuracies, f1_scores, sensitivities, specificities, roc_aucs = [], [], [], [], []

# Counters for calculating specificity
true_negatives = 0
false_positives = 0

# Message counter
count = 0

# Track the time of the last message
last_message_time = time.time()
timeout = 30  # 2-minute timeout

# Initialize the Kafka consumer
consumer = KafkaConsumer(
    topic_name,
    bootstrap_servers=bootstrap_servers,
    auto_offset_reset="earliest",
    group_id="deep-river-consumer-group",
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),  # Deserialize JSON string to dict
    key_deserializer=lambda x: x.decode('utf-8')  # Decode key to string
)


# Function to process each Kafka message
def process_message(key, value):
    global count, true_negatives, false_positives
    # Key represents the label, value is the feature dictionary
    y_true = int(key)  # Convert key (output label) to integer

    # Convert all feature values to floats
    features = {k: float(v) for k, v in value.items()}
    # Make a prediction
    y_pred = model.predict_one(features)

    y_pred_proba = model.predict_one(features)


    # Update metrics if a prediction is made
    if isinstance(y_pred_proba, (int, float)):
        y_pred = 1 if y_pred_proba >= 0.5 else 0
        accuracy.update(y_true, y_pred)
        f1_score.update(y_true, y_pred)
        sensitivity_metric.update(y_true, y_pred)
        roc_auc.update(y_true, y_pred)

        # Calculate true negatives and false positives for specificity
        if y_true == 0 and y_pred == 0:
            true_negatives += 1
        elif y_true == 0 and y_pred == 1:
            false_positives += 1

    # Calculate specificity if there are true negatives and false positives
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    # Store metrics every 50 messages
    count += 1
    if count % 3000 == 0 or count>14545:
        print(f"Processed {count} messages")
        print(f"Latest value processed: {value}")  # Print the latest value
        accuracies.append(accuracy.get() * 100)
        f1_scores.append(f1_score.get() * 100)
        sensitivities.append(sensitivity_metric.get() * 100)
        specificities.append(specificity * 100)
        roc_aucs.append(roc_auc.get() * 100)

        # Print current metrics
        print(f"Accuracy: {accuracies[-1]:.2f}%")
        print(f"F1 Score: {f1_scores[-1]:.2f}%")
        print(f"Sensitivity: {sensitivities[-1]:.2f}%")
        print(f"Specificity: {specificities[-1]:.2f}%")
        print(f"ROC AUC: {roc_aucs[-1]:.2f}%")


    model.learn_one(features, y_true)

# Function to plot metrics
def plot_metrics():
    plt.figure(figsize=(10, 6))

    # Plot each metric over time
    plt.plot(accuracies, label="Accuracy")
    plt.plot(f1_scores, label="F1 Score")
    plt.plot(sensitivities, label="Sensitivity")
    plt.plot(specificities, label="Specificity")
    plt.plot(roc_aucs, label="ROC AUC")

    # Labeling
    plt.xlabel("Batches of 50 Messages")
    plt.ylabel("Percentage")
    plt.title("Model Performance Metrics Over Time")
    plt.legend()
    plt.show()


# Consume and process messages from Kafka
try:
    print("Starting Kafka Consumer...")
    while True:
        # Poll for new messages with a short timeout
        msg_pack = consumer.poll(timeout_ms=1000)  # 1-second poll timeout

        # Check if any messages were received
        if msg_pack:
            for tp, messages in msg_pack.items():
                for message in messages:
                    # Update the last message time
                    last_message_time = time.time()

                    # Access the deserialized key and value
                    key = message.key  # This is the 'output' column from the CSV, representing the label
                    value = message.value  # This is the feature dictionary without the 'output' column

                    # Process the message
                    process_message(key, value)

        # Check if no message has been received for the specified timeout
        if time.time() - last_message_time >= timeout:
            print("No messages received for 2 minutes. Plotting final metrics and closing consumer.")
            plot_metrics()
            break  # Exit the loop if timeout occurs

except KeyboardInterrupt:
    print("Consumer interrupted manually. Closing...")

except Exception as e:
    print(f"Error processing message: {e}")

finally:
    # Optional: Close the consumer if you want to do cleanup
    consumer.close()
