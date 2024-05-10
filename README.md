# Detection-of-Common-Lung-Diseases-Through-Chest-X-ray-Imaging

This respoitory contains code to detect common lung deseases using the NIH and Covid19 datasets from Kaggle. In this respoistory, contains the 4 model architecture (EfficientNetB2, DenseNet169, Alexnet, and VGG16), the evaluation script, the models we trained for 50 epochs, and the combind dataset of NIH and Covid-19 datasets.

#### Requirements
- Python 	3.11
- pytorch
- matplotlib
- numpy
- pandas

### Datasets
- NIH Chest X-Rays (https://www.kaggle.com/datasets/nih-chest-xrays/data)
- Covid-19 Radiograph Database (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

### Reproducibility
To reprocude our results, simply run any of the model train python files using python3 <model>.py
This will result in the training of said model for 50 epochs and once finished, it will produce 2 graphs, one being the train and validation loss over epochs and the other will be accuracy over epochs.

To repoduce our results of our evaluation metrics, simply replace the name of the model you wish to evaluate when loading the model, and change the architecture in the code.

