from kafka import KafkaProducer
import csv
import json
import random
from time import sleep

# Kafka configurations
kafka_broker = "localhost:9092"
topic = "MI-river"
bootstrap_servers = "localhost:9092"

# Paths to the CSV files to read data from
csv_file_paths = [
    '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_abnormal_20.csv',
    '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_abnormal_80_label.csv',
    '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_normal_20.csv',
    '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_normal_80_label.csv'
]

def load_and_shuffle_data(files):
    all_rows = []
    header = None

    # Read and concatenate data from each file
    for csv_file in files:
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            file_header = next(csv_reader)  # Get the header for each file

            # Ensure the headers are consistent across all files
            if header is None:
                header = file_header
            elif header != file_header:
                raise ValueError("Headers do not match across all files.")

            # Collect all rows
            all_rows.extend(list(csv_reader))

    # Shuffle all rows together after concatenation
    random.shuffle(all_rows)
    return header, all_rows

def produce_json_messages(producer, header, rows):
    # Find the index of the 'output' column
    output_index = header.index("label")

    for row in rows:
        # Convert row to a dictionary
        message = dict(zip(header, row))

        # Use the 'output' column as the key
        key = row[output_index]  # Get the 'output' column value

        # Remove the 'output' column from the message
        message.pop("label")

        # Send the message with both key and value
        producer.send(topic, key=key.encode('utf-8'), value=message)
        print(f"Sending data to topic '{topic}' with key '{key}': {message}")
        sleep(0.001) # Sleep to control message sending rate

if __name__ == "__main__":
    # Create a Kafka producer instance with JSON serializer for values
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load and shuffle data from all files
    header, all_rows = load_and_shuffle_data(csv_file_paths)

    # Produce messages from the concatenated and shuffled data
    produce_json_messages(producer, header, all_rows)

    # Close the producer after sending all messages
    producer.close()
