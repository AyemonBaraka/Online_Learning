import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Set the Kafka bootstrap server
bootstrap_servers = "localhost:9092"
test_file = '/home/ayemon/KafkaProjects/kafkaspark07_90/mitbih_train.csv'

# Load your test dataset
col_names = [f't{i}' for i in range(1, 188)] + ['target']
test_df = pd.read_csv(test_file, header=None)
test_df.columns = col_names

# Shuffle the dataset and binarize the target column
shuffled_df = test_df.sample(frac=1).reset_index(drop=True)
shuffled_df['target'] = shuffled_df['target'].apply(lambda x: 1 if x != 0 else 0)

# Kafka configuration
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

def write_to_kafka(topic_name, items):
    count = 0
    for message, key in items:
        try:
            producer.send(topic_name, key=key.encode('utf-8'), value=message.encode('utf-8'))
            count += 1
        except KafkaError as e:
            print(f"Error sending message to Kafka: {e}")

    producer.flush()
    print("Wrote {0} messages into topic: {1}".format(count, topic_name))

# Prepare test data
x_test_df = shuffled_df.drop(["target"], axis=1)
y_test_df = shuffled_df["target"]

x_test = x_test_df.values.tolist()
y_test = y_test_df.values.tolist()

# Sending test data to Kafka
write_to_kafka("susy-test", zip(map(lambda x: ','.join(map(str, x)), x_test), map(str, y_test)))

# Close the producer connection after sending the data
producer.close()
