# Iris Flower Species Classification
This project is a machine learning application that classifies Iris flowers into one of three species: Iris-setosa, Iris-versicolor, and Iris-virginica. The classification is based on the measurements of the flower's sepals and petals.

## Project Overview
The dataset used in this project consists of five columns:

```sepal_length
sepal_width
petal_length
petal_width
```
## species (the target variable)
The species column contains the names of the species, which will be predicted based on the other four features.

## Steps Involved
 1 Data Loading and Preprocessing:

The Iris dataset is loaded from a CSV file.
The species names are converted to numerical labels for model training.
The dataset is split into training and testing sets.

2 Model Training:

A Random Forest classifier is trained on the training data.

3 Model Evaluation:

The trained model is evaluated on the test set using accuracy, confusion matrix, and classification report.

4 Visualization:

Various visualizations, including pair plots, confusion matrix heatmap, and decision boundary plots, are created to better understand the data and model performance.

5 Prediction:

The trained model is used to predict the species of a new Iris flower based on sepal and petal measurements.

## Visualizations

The project includes the following visualizations:

Pair Plot: Shows the relationships between the four features, colored by species.


Confusion Matrix Heatmap: Displays the performance of the classifier in distinguishing between the different species.


Decision Boundary Plot: Visualizes the decision boundary of the classifier (for a subset of features).


Requirements
```
Python 3.x
pandas
scikit-learn
seaborn
matplotlib
```
You can install the required packages using pip:
```
Copy code
pip install pandas scikit-learn seaborn matplotlib
```
Running the Project
Clone the repository:
```
Copy code
git clone https://github.com/yourusername/iris-flower-classification.git
```
Navigate to the project directory:
```
Copy code
cd iris-flower-classification
```
Run the script:
```
Copy code
python iris_classification.py
```
This will load the dataset, train the model, visualize the results, and make a prediction.

## Example Prediction:

The script includes an example input:
```
python
Copy code
example = [[5.1, 3.5, 1.4, 0.2]]
```
The prediction will output the species name:
Copy code
```
Predicted Species: Iris-setosa
```
Project Structure
iris_classification.py: The main script containing all the steps from data loading to prediction.

iris.csv: The dataset used for training and evaluation.

README.md: This file.

## Contributing
If you want to contribute to this project, feel free to fork the repository and submit a pull request.
