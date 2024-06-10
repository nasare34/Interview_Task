# Interview_Task
DOCUMENTATION

1. Methodology
Data Generation: - Constructed a tiny dataset comprising actual products' names, descriptions, prices, and categories.

2. 
Data Preprocessing
•	a.Using the Keras Tokenizer, tokenized the name and description of the text characteristics.
•	b. Sequences with padding to guarantee even length.
•	c. Used LabelEncoder to encode the category, or categorical attribute.

3. Model Development
TensorFlow was used to create a neural network model.
•	Used layers for embedding, flattening, and densification.
•	Used the Adam optimizer and sparse categorical cross-entropy loss to compile the model.

4. Training and Evaluation of the Model: 
•	The model was trained using the complete dataset, and its performance was assessed on the training and test sets.
•	Showed strong generalization with an accuracy of 75% on the training set and 100% on the test set.

Assumptions: 
The synthetic, tiny dataset is a condensed representation of real-world data.
It is expected that the text properties are brief and uncomplicated, which makes tokenization and padding simple. Without requiring a lot of tweaking, the model architecture and hyperparameters were selected for ease of use and speed of development.

Limitations: 
Because of the simplicity and limited size of the dataset, the model's performance measures could be biased. The attained accuracy may not be representative of performance in the real world, particularly in more intricate situations involving more datasets.
For best results in practical applications, more testing with various model topologies, hyperparameters, and data pretreatment methods would be required.


	INSTRUCTIONS FOR RUNNING THE PROJECT

For running the code, follow these steps:
1.	Make sure you have Python installed on your system along with the required libraries (TensorFlow, Pandas, NumPy ).
2.	import numpy as np
3.	from sklearn.preprocessing import LabelEncoder
4.	from tensorflow.keras.preprocessing.text import Tokenizer
5.	from tensorflow.keras.preprocessing.sequence import pad_sequences
6.	import pandas as pd
7.	import tensorflow as tf
8.	from tensorflow.keras.models import Sequential
9.	from tensorflow.keras.layers import Embedding, Flatten, Dense
10.	from sklearn.model_selection import train_test_split
11.	

12.	Open a text editor and copy the code provided in the Python script or Jupyter Notebook.
13.	Save the file with a .py extension if it's a Python script or .ipynb extension if it's a Jupyter Notebook. (Interview_Task.ipynb or Interview_Task.py)
14.	Open a terminal or command prompt and navigate to the directory where you saved the file.
15.	Run the Python script using the following command:
python Interview_Task.ipynb.py
After running the code, you should have the key results and outputs generated during the execution of your code

screenshots of key results and outputs generated during the execution of your code.


  
![image](https://github.com/nasare34/Interview_Task/assets/83691115/ae86b33a-7e7a-4cb1-aa31-9663b94aea68)
