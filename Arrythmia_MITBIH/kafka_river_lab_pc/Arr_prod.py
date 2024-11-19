from kafka import KafkaProducer
import csv
import json
import random
from time import sleep

# Kafka configurations
kafka_broker = "localhost:9092"
topic = "Arr-river"
bootstrap_servers = "localhost:9092"

# Paths to the CSV files to read data from
csv_file_paths = [
    '/home/ayemon/KafkaProjects/kafkaspark07_90/mitbih_test.csv',
    '/home/ayemon/KafkaProjects/kafkaspark07_90/mitbih_train.csv',
]

def load_and_shuffle_data(files):
    all_rows = []

    # Read and concatenate data from each file
    for csv_file in files:
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            # Collect all rows
            all_rows.extend(list(csv_reader))

    # Shuffle all rows together after concatenation
    random.shuffle(all_rows)
    return all_rows

def produce_json_messages(producer, rows):
    for row in rows:
        # Use the last column as the key
        raw_key = float(row[-1])  # Parse the last column as a float
        key = "0" if raw_key == 0 else "1"  # Transform the key
        #key = int(float(key))

        # Use the remaining columns as the message value
        message = {f"feature_{i}": value for i, value in enumerate(row[:-1])}

        # Send the message with both key and value
        producer.send(topic, key=key.encode('utf-8'), value=message)
        print(f"Sending data to topic '{topic}' with key '{key}': {message}")
        # sleep(0.001) # Uncomment to control message sending rate

if __name__ == "__main__":
    # Create a Kafka producer instance with JSON serializer for values
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load and shuffle data from all files
    all_rows = load_and_shuffle_data(csv_file_paths)

    # Produce messages from the concatenated and shuffled data
    produce_json_messages(producer, all_rows)

    # Close the producer after sending all messages
    producer.close()
