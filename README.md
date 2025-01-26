# Deep-Learning--Waste-Classification
Using a CNN DL model to classify images of waste into biodegradable or not

# Overview 
This project uses deep learning to classify waste into 8 subcategories using a Convolutional Neural Network (CNN) based on the Xception pre-trained model. Then, the waste is grouped into two main categories: biodegradable and non-biodegradable. By automating waste classification, this model seeks to contribute to more efficient waste management and promote sustainability.

## Why This Problem Matters
Pollution is one of the most pressing challenges of our time, with waste mismanagement contributing increasingly to environmental degradation. Improper disposal of non-biodegradable materials leads to long-lasting harm, such as soil contamination, water pollution, and harm to wildlife. Additionally, failing to identify and process biodegradable waste properly can increase greenhouse gas emissions like methane.

By accurately classifying waste, this project can help:
- Enhance recycling efforts by correctly sorting waste
- Reduce landfill overflow by ensuring proper waste management
- Support sustainability initiatives by promoting the reuse and recycling of materials
- Educate individuals and organizations on the importance of waste segregation
  Through the application of advanced AI technologies like CNNs, this project aims to bridge the gap between waste generation and sustainable waste management, ultimately contributing to a cleaner and greener planet


# Dataset 
Kaggle : https://www.kaggle.com/datasets/aashidutt3/waste-segregation-image-dataset/data

# Steps
- Data Exploration, visualization, and preprocessing, including handling missing values, scaling data, and feature selection
- Training the Xception model with a new top, fine-tuning different parameters like the learning rate, inner layer size and dropout, and analyzing performance
- Training a larger model and using it to make predictions
- Transforming to tf-lite, removing tensorflow dependenciesn for a smaller model and ease of deployment
- Deploying the best model as a web service using Waitress
- Containerization of the application with Docker

# Main libraries Used
numpy, matplotlib, tensorflow, keras, tf-lite, 



 # Prerequisites
- Python 3.7 or above
- Docker (if you wish to run the application in a container).<br>
- pipenv for setting the environment. You can install it using:
```bash
pip install pipenv 
```

You can check if Python and Docker are already installed on your machine by running the following commands in your terminal:
```bash
python --version
```
```bash
docker --version
```


  
# Project Structure
This repository contains:
- **notebook.ipynb**: Jupyter Notebook containing the exploratory data analysis (EDA), data preprocessing, different models training, and evaluation steps.
- **train.py**: Script to train the final and best machine learning model.
- **predict.py**: Script for making predictions with the trained model.
- **predict-test-request.py**: Script to test prediction requests.
- **model.bin**: The trained model, stored in binary format.
- **load_model_test.py**: A test script to load and validate the model from water_model.bin.
- **Pipfile**: Specifies the Python packages required to run the project.
- **Pipfile.lock**: Contains exact versions of dependencies as installed in the virtual environment.
- **Dockerfile**: The Docker configuration file. Used to containerize the application for consistent deployment.



# Installation

1. Clone the repository:
2. Set up a virtual environment and install the required dependencies:
   ```bash
   pipenv install

3. (Optional) To run the project inside Docker, build the Docker image:
   ```bash
   docker build -t water-potability-prediction . !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

4. (Optional) Run the Docker container:
   ```bash
   docker run -p 9696:9696 water-potability-prediction    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Running the Scripts
### Training the Model

To train the machine learning model, run:
```bash
python train.py
```
This script will preprocess the data, train a model, and save it as water_model.bin.

### Making Predictions
Once the model is trained, you can make predictions on new data by running the predict.py script:
```bash
python predict.py
```

You can also test the prediction process by using the predict-test-request.py script.

### Testing the Model
Use load_model_test.py to load and test the model:
```bash
python load_model_test.py
```

# Jupyter Notebook
For detailed data exploration and visualization, different model training, parameter tuning, and choosing the final model, open the ``notebook.ipynb`` notebook. You can run the notebook locally in Jupyter.

# Docker Usage

If you prefer to run the application inside Docker, follow these steps:

- Build the Docker image:
```bash
docker build -t water-potability . !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
```

- Run the Docker container:
```bash
docker run -p 9696:9696 water-potability
```

This will start a local server on port 9696, and you can make requests to the application.

# License

This project is licensed under the MIT License - see the LICENSE file for details.


### Final Note

> Thank you for checking out this project! 
> Contributions are welcome. Feel free to fork this repository, create a pull request, or raise an issue.
