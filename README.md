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
<br>PS: I have installed it on my local machine while creating this project, and I can't commit it because it is a huge file.<br>

### **Automatic Download**
The dataset will be downloaded automatically when running the script.  
Ensure you have the **Kaggle API key** set up:

1. **Get Your API Key**  
   - Sign in to [Kaggle](https://www.kaggle.com/)  
   - Go to [Account Settings](https://www.kaggle.com/account)  
   - Scroll to **API** → Click **Create New API Token**  
   - This downloads a `kaggle.json` file  

2. **Place the API Key**  
   - Move `kaggle.json` to the correct location:  
     ```sh
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Run the Training Script**  
   ```sh
   python train.py
   ```


# Steps
- Data Exploration, visualization, and preprocessing, including handling missing values, scaling data, and feature selection
- Training the Xception model with a new top, fine-tuning different parameters like the learning rate, inner layer size and dropout, and analyzing performance
- Training a larger model and using it to make predictions
- Deploying the best model as a web service using Flask and Waitress
- Containerization of the application with Docker

# Main libraries Used
numpy, matplotlib, tensorflow, keras. 



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
- **notebook-Xception.ipynb**: Jupyter Notebook containing the exploratory data analysis (EDA), data preprocessing, different models training, and evaluation steps.
- **train.py**: Script to train the final and best machine learning model.
- **predict.py**: Script for making predictions with the trained model, contains the flask app.
- **xception_299_04_0.934.keras**: The best trained model, stored in binary format.
- **Pipfile**: Specifies the Python packages required to run the project.
- **Pipfile.lock**: Contains exact versions of dependencies as installed in the virtual environment.
- **Dockerfile**: The Docker configuration file. Used to containerize the application for consistent deployment.
- **index.html**: provides a user interface at http://localhost:9696/ where you can upload an image and get a result
- **download_data.py** : to download the Kaggle dataset using the Kaggle API



# Installation

1. Clone the repository:
2. Set up a virtual environment and install the required dependencies:
   ```bash
   pipenv install

3. (Optional) To run the project inside Docker, build the Docker image:
   ```bash
    docker build -t waste-segregation-app .  
4. (Optional) Run the Docker container:
   ```bash
   docker run -p 9696:9696 waste-segregation-app
   
# Running the Scripts
### Training the Model

To train the machine learning model, run:
```bash
python train.py
```
This script will preprocess the data, train a model, and save it as xception{epoch}-{accuracy}.keras (saving best model only)

### Making Predictions

1. Once the model is trained, you can make predictions on new data by running the predict.py script:
```bash
python predict.py
```

2. The Flask server starts running.<br>
  Open your browser, go to : http://localhost:9696/ to see the webpage interface

3. Upload an image to see the result


# Docker Usage

If you prefer to run the application inside Docker, follow these steps:
- Download the [dataset](https://www.kaggle.com/datasets/aashidutt3/waste-segregation-image-dataset/data) from Kaggle 

- Build the Docker image:
```bash
docker build -t waste-classifier .  
```
- Mount the dataset into the container:
```bash
docker run -v /path/to/dataset:/app/waste-segregation-dataset my-deep-learning-project
```

- Run the Docker container:
```bash
docker run -p 9696:9696 waste-classifier
```

This will start a local server on port 9696<br>
Then, in a new terminal window, you can test it using curl, by providing the path to an image you have:<br>

- Test with an image:
```bash
curl -X POST -F "file=@path_to_your_image.jpg" http://localhost:9696/predict
```
Replace /path/to/your/image.jpg with the actual path to your test image.



# Jupyter Notebook
For detailed data exploration and visualization, different model training, parameter tuning, and choosing the final model, open the ``notebook.ipynb`` notebook. You can run the notebook locally in Jupyter.

# License

This project is licensed under the MIT License - see the LICENSE file for details.


### Final Note

> Thank you for checking out this project! 
> Contributions are welcome. Feel free to fork this repository, create a pull request, or raise an issue.
