# ChurnAnalysis
This project contains all of the necessary libraries for the customer churn project. The dataset_creator file is used to create synthetic data, based on real uploaded customer data. The main.py file is used to generate predictions, utilizing the synthetic data as a base. 

To generate synthetic data, data must be fed into either the dataset_creator or wgan_gp file. There is a sample dataset provided. This can be uploaded into the dataset_creator or the wgan_gp to create more synthetic data to create a training set for the main churn analysis classification model. 

The dataset_creator file will augment and label real customer data, replicating the trends and format of the highest-quality data available. The wgan_gp file runs a WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty) to create high-quality, purely synthetic data. Due to the computationally expensive nature of GANs, I recommend you create datasets comprised of a 5:1 ratio of data from dataset_creator and wgan_gp. 

For more information on WGAN-GP and how it affects the model's output, I recommend reading the attached research pdf. 
