import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, recall_score
from tensorflow.keras.optimizers import SGD
# Load your dataset
csv_file_paths = [
    '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_abnormal_20.csv',
    '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_abnormal_80_label.csv',
    '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_normal_20.csv',
    '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_normal_80_label.csv'
]


def load_data(files):
    dataframes = [pd.read_csv(file) for file in files]  # No need for header=None since the files have headers
    data = pd.concat(dataframes, ignore_index=True)
    return data


data = load_data(csv_file_paths)
data.dropna(inplace=True)
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values  # Last column as the target label

# Convert labels to categorical (one-hot encoding) for multi-class classification
num_classes = len(set(y))  # Number of unique classes
y = to_categorical(y, num_classes=num_classes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a more complex neural network model
model = Sequential([
    Dense(512, input_shape=(X_train.shape[1],), activation='elu'),
    Dropout(0.1),

    Dense(256, activation='elu'),  # Second dense layer with L2 regularization
    Dropout(0.1),  # Dropout layer with 30% dropout rate

    Dense(128, activation='elu'),  # Second dense layer with L2 regularization

    Dense(64, activation='elu'),  # Third dense layer with L2 regularization
    Dropout(0.1),  # Dropout layer with 30% dropout rate

    Dense(32, activation='elu'),  # Fourth dense layer with L2 regularization
    #Dropout(0.2),  # Dropout layer with 20% dropout rate

    Dense(num_classes, activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adam optimizer with a custom learning rate
    loss='binary_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Metrics to track during training
)
"""
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # SGD with momentum and Nesterov
model.compile(
    optimizer=sgd,                          # Using SGD optimizer
    loss='binary_crossentropy',        # Loss function for multi-class classification
    metrics=['accuracy']                    # Metrics to track during training
)

"""
# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitors validation loss
    patience=20,  # Number of epochs with no improvement after which training will stop
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

# Train the model with early stopping
history = model.fit(
    X_train, y_train,
    epochs=150,  # Maximum number of epochs
    batch_size=32,  # Batch size
    validation_split=0.2,  # Use 20% of training data for validation
    callbacks=[early_stopping],  # Add early stopping to callbacks
    verbose=1  # Print progress during training
)

# Predict on the test data
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)        # Convert probabilities to class labels
y_test_classes = y_test.argmax(axis=1)        # Convert one-hot encoded labels back to class labels

# Accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)

# F1 Score (macro average for multi-class)
f1 = f1_score(y_test_classes, y_pred_classes, average='macro')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
tn, fp, fn, tp = conf_matrix.ravel() if conf_matrix.shape == (2, 2) else (0, 0, 0, 0)

# Sensitivity (Recall)
sensitivity = recall_score(y_test_classes, y_pred_classes, average='macro')  # Average sensitivity for multi-class

# Specificity
if conf_matrix.shape == (2, 2):
    specificity = tn / (tn + fp)  # Specificity calculation for binary classification
else:
    specificity = None  # Specificity calculation requires binary classes

# ROC AUC (macro average for multi-class)
roc_auc = roc_auc_score(y_test, y_pred, average='macro', multi_class='ovr')

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Macro): {f1:.4f}")
print(f"Sensitivity (Recall, Macro): {sensitivity:.4f}")
print(f"Specificity: {specificity if specificity is not None else 'Not applicable for multi-class'}")
print(f"ROC AUC (Macro): {roc_auc:.4f}")

print("\nClassification Report:\n", classification_report(y_test_classes, y_pred_classes))



Accuracy: 0.9835
F1 Score (Macro): 0.9795
Sensitivity (Recall, Macro): 0.9806
Specificity: 0.9740420271940667
ROC AUC (Macro): 0.9944

Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.97      0.97       809
           1       0.99      0.99      0.99      2101

    accuracy                           0.98      2910
   macro avg       0.98      0.98      0.98      2910
weighted avg       0.98      0.98      0.98      2910
