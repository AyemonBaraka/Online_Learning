from kafka import KafkaProducer
import csv
from time import sleep
import json

# Kafka broker address
kafka_broker = "localhost:9092"
# Kafka topic to produce messages to
topic = "ha-river"
# Kafka broker address
bootstrap_servers = "localhost:9092"
# Path to the CSV file to read data from
csv_file_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/heart.csv"


# Function to read the CSV file and send rows as messages to Kafka
def produce_json_messages(producer, csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Get the header

        # Find the index of the 'output' column
        output_index = header.index("output")

        for row in csv_reader:
            # Convert row to a dictionary
            message = dict(zip(header, row))

            # Use the 'output' column as the key
            key = row[output_index]  # Get the 'output' column value

            # Remove the 'output' column from the message
            message.pop("output", None)

            # Send the message with both key and value
            producer.send(topic, key=key.encode('utf-8'), value=message)
            print(f"Sending data to topic '{topic}' with key '{key}': {message}")
            sleep(0.10)  # Sleep to control message sending rate


if __name__ == "__main__":
    # Create a Kafka producer instance with JSON serializer for values
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Produce messages from the CSV file
    produce_json_messages(producer, csv_file_path)

    # Close the producer after sending all messages
    producer.close()
