# MovieSuccessPrediction
Capstone Project (Movie_Success_Prediction)
This Streamlit app predicts the success of a movie based on various features using a pre-trained Random Forest classifier model.

Key Components:

Load Model and Data:

Loads the trained Random Forest classifier model (movie_classifier.pkl).
Loads the feature data (movie_features.csv) for creating dropdown options.
Loads the original dataset (movie_data_full.csv) for mapping categorical values.
Define Categorical Columns and Mappings:

Specifies the categorical columns (categorical_cols_X) for label encoding.
Creates a dictionary (label_encoding_mappings) to store the mapping between original and encoded values for categorical features.
Creates a dictionary (budget_duration_mappings) to store the mapping between scaled and original values for Budget and Duration of the Movie.
Create Input Widgets:

Creates dropdown options for categorical features using original values from the original dataset.
Creates a number input for Duration of the Movie with a range from 1 to 5 hours.
Creates a number input for Budget with a specified range and step size.
Handle User Input:

Collects user input from the dropdown and input fields.
Maps categorical values back to encoded values using the label_encoding_mappings dictionary.
Maps Budget and Duration of the Movie back to scaled values using the budget_duration_mappings dictionary.
Make Prediction:

Creates an input DataFrame with the collected user input.
Preprocesses the input data using the same steps as the training data.
Makes a prediction using the trained Random Forest classifier model.
Display Prediction:

Displays the predicted movie success category (Flop, Average, or Hit) based on the model's output.
Key Features:

User-friendly interface: Provides a simple and intuitive interface for users to input movie features and get predictions.
Accurate predictions: Leverages a pre-trained Random Forest classifier model for accurate predictions.
Data preprocessing: Handles categorical and numerical features, including label encoding and scaling.
Customizable input: Allows users to input various movie features to tailor the prediction.
Clear output: Displays the predicted movie success category in a user-friendly format.
This Streamlit app effectively demonstrates the capabilities of the trained model and provides a valuable tool for predicting movie success based on the given features.
