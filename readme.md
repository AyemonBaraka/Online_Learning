![image](https://github.com/AyemonBaraka/Online_Learning/assets/123589496/bdfe67a5-2023-407b-bc48-601326b7cb92)


Conventional Machine learning model works with batch data. To train the model we need all the data at once.

If we want to work with new data, we need to add the previous data with new data and train the model from scratch.

Incremental Learning help us from this. There is a library called **river**, that is used in online machine learning. It can work with new data. It keep training the model when new data is available.


## The River API

River is a python library for online machine learning. The machine learning model in river can learn from one data. When the new data arrives, model can be learned from the new data and updated instantly.

The APIs in river that are usually used is given below:

**learn_one(x, y):** This API update the model when the single instance containing input features x and target value y arrives.

**predict_one(x):** (classification, regression, clustering) returns the prediction from model for a single observation

**predict_proba_one(x):** (classification) returns the model’s predicted probability for a single observation

**score_one:** (anomaly detection) returns the outlier score for a single observation

**transform_one:** transforms one input observation


## Heart Attack Classification using Online Machine Learning

In this work heart attack is classified using online machine learning. When the heart attack data arrives, the model learns instantly. The datase contains the information given below:

![image](https://github.com/AyemonBaraka/Online_Learning/assets/123589496/57d7aadd-3087-4f01-969e-87c0bd7ef058)

### Online Machine Learning

#### Model
Here, logistic regression model is used.

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
    )
    
#### Training Model

'def train(X, y):
    #pipeline = get_pipeline()

    # Initialize metrics

    f1_score = metrics.MacroF1()
    cm = metrics.ConfusionMatrix()
    accuracy = metrics.Accuracy()
    f1_scores = []
    accuracies = []

    # Iterate over the dataset
    for xi, yi in stream.iter_pandas(X, y, shuffle=True, seed=1):
        # Predict the new sample
        yi_pred = model.predict_one(xi)

        # Get the score
        if yi_pred is not None:
            f1_score.update(yi, yi_pred)
            f1_scores.append(f1_score.get() * 100)

            cm.update(yi, yi_pred)

            accuracy.update(yi, yi_pred)  # update the accuracy
            accuracies.append(accuracy.get() * 100)

        # Train the model with the new sample
        model.learn_one(xi, yi)
    #print(accuracy)
    return accuracies, f1_scores, cm, model'


In this train function, the model is updated as soon as the one new data arrives.

#### Result
The Accuracy is given below for every 30 data.

    [30]  Accuracy: 80.00%
    [60]  Accuracy: 83.33%
    [90]  Accuracy: 85.56%
    [120] Accuracy: 85.83%
    [150] Accuracy: 85.33%
    [180] Accuracy: 85.00%
    [210] Accuracy: 83.81%
    [240] Accuracy: 83.75%
    [242] Accuracy: 83.88%

The final Accuracy is 83.88%


The plot of F1_score for every data is given below.

![image](https://github.com/AyemonBaraka/Online_Learning/assets/123589496/4a30a760-8be7-4a8b-bfde-18f5381ee010)

The plot of Accuracy for every data is given below.

![image](https://github.com/AyemonBaraka/Online_Learning/assets/123589496/23cbfa6d-82a5-471d-b0f3-79d250b28a65)
