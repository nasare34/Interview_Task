# Interview_Task Documentation

## Methodology

### Data Generation:
- Constructed a tiny dataset comprising actual products' names, descriptions, prices, and categories.

### Data Preprocessing:
- Utilized the Keras Tokenizer to tokenize the text characteristics of names and descriptions.
- Padded sequences to ensure even length.
- Employed LabelEncoder to encode the categorical attribute.

### Model Development:
- Developed a neural network model using TensorFlow.
- Incorporated layers for embedding, flattening, and densification.
- Compiled the model using the Adam optimizer and sparse categorical cross-entropy loss.

### Training and Evaluation of the Model:
- Trained the model using the complete dataset and assessed its performance on the training and test sets.
- Demonstrated strong generalization with an accuracy of 75% on the training set and 100% on the test set.

## Assumptions:
- The synthetic, tiny dataset serves as a condensed representation of real-world data.
- Text properties are assumed to be brief and uncomplicated, simplifying tokenization and padding.
- The model architecture and hyperparameters were selected for ease of use and speed of development without extensive tweaking.

## Limitations:
- Due to the simplicity and limited size of the dataset, the model's performance measures could be biased.
- The attained accuracy may not be representative of performance in real-world scenarios, especially in more intricate situations involving larger datasets.
- Further testing with various model topologies, hyperparameters, and data pretreatment methods would be required for optimal performance in practical applications.

## Instructions for Running the Project

### Prerequisites:
- Ensure Python is installed on your system along with the required libraries (TensorFlow, Pandas, NumPy).

### Steps:
1. Open a text editor and copy the code provided in the Python script or Jupyter Notebook.
2. Save the file with a `.py` extension if it's a Python script or `.ipynb` extension if it's a Jupyter Notebook (e.g., `Interview_Task.ipynb` or `Interview_Task.py`).
3. Open a terminal or command prompt and navigate to the directory where you saved the file.
4. Run the Python script using the following command:

```bash
python Interview_Task.ipynb.py
```

5. After running the code, you should have key results and outputs generated during the execution of your code.

## Screenshots
![Key Results and Outputs](https://github.com/nasare34/Interview_Task/assets/83691115/ae86b33a-7e7a-4cb1-aa31-9663b94aea68)
