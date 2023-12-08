import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from gensim.models import FastText

class RegressionDataFrameDataset(Dataset):
    def __init__(self, dataframe, type_encoder, is_train=True):
        self.data = dataframe
        self.data.drop('_id', axis=1, inplace=True)
        self.data.drop('Deposit', axis=1, inplace=True)
        self.data.drop('Location', axis=1, inplace=True)
        self.embeddings_model = FastText.load("models/fastText_model")

        # Apply label encoding to 'Floor' column
        self.data['Floor'] = self.data['Floor'].apply(self.extract_floor_number)

        # Convert specified boolean columns to integers
        boolean_columns = ['Storage room', 'Wardrobe', 'Furnished', 'Equipped kitchen', 'Renovation', 
                           'Reduced mobility', 'Heating', 'Garage', 'Elevator', 'Air conditioning',
                            'Swimming pool', 'Garden', 'Green areas', 'Terrace']
        self.data[boolean_columns] = self.data[boolean_columns].astype(int)

        # One-hot encoding for 'Type' column
        self.type_encoder = type_encoder
        types_encoded = self.type_encoder.transform(self.data[['Type']])

        # Convert 'Description' to embeddings
        self.descriptions = self.convert_to_embeddings(self.data['Description'])

        # Combine all features
        self.features = np.hstack([types_encoded, self.descriptions, self.data.drop(columns=['Type', 'Description', 'Price'])])

        self.targets = self.data['Price'].values

    def convert_to_embeddings(self, descriptions):
        embeddings = []

        for desc in descriptions:
            # Split the description into words and filter out empty words
            words = [word for word in desc.split() if word]

            # Get embeddings for each word and average them
            word_embeddings = [self.embeddings_model.wv[word] for word in words if word in self.embeddings_model.wv]
            if word_embeddings:
                desc_embedding = np.mean(word_embeddings, axis=0)
            else:
                # Handle case with no words in model's vocabulary
                desc_embedding = np.zeros(self.embeddings_model.vector_size)

            embeddings.append(desc_embedding)

        return np.array(embeddings)
        # return np.random.randn(len(descriptions), 128)
    
    @staticmethod
    def extract_floor_number(floor):
        if 'entreplanta' in floor:
            return -100
        if 'planta' in floor:
            # Extraer y devolver el número de planta
            floor_number = floor.split()[1]
            floor_number = ''.join(filter(str.isdigit, floor_number))
            return int(floor_number)
        elif 'bajo' in floor:
            return 0
        elif 'semi-sótano' in floor:
            return -1
        elif 'sótano' in floor:
            return -2
        elif 'exterior' in floor or 'interior' in floor:
            return -999
        elif 'no aplica' in floor:
            return 999
        else:
            print('Value of floor which has not been contemplated: ' + floor)
            return None 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.targets[idx], dtype=torch.float)

class RegressionDataModule(LightningDataModule):
    def __init__(self, dataframe, batch_size=32):
        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size
        # - handle_unknown='ignore' tells the encoder to ignore any categories in the validation/test set that were not present in the training set
        # - sparse_output=False to return a numpy array instead of a sparse matrix
        self.type_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

    def setup(self, stage=None):
        # Split data into train, validation, and test sets
        train_val_df, test_df = train_test_split(self.dataframe, test_size=0.2)
        train_df, val_df = train_test_split(train_val_df, test_size=0.25) # 0.25 x 0.8 = 0.2

        # Fit the OneHotEncoder here using only the training data
        self.type_encoder.fit(train_df[['Type']])

        self.train_dataset = RegressionDataFrameDataset(train_df, self.type_encoder, is_train=True)
        self.val_dataset = RegressionDataFrameDataset(val_df, self.type_encoder, is_train=False)
        self.test_dataset = RegressionDataFrameDataset(test_df, self.type_encoder, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('data/preprocessed_data.csv')
    data_module = RegressionDataModule(df, batch_size=64)
    data_module.setup()

    for batch in data_module.train_dataloader():
        x, y = batch
        print(x.shape, y.shape)
