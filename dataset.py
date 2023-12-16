import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
import numpy as np
from gensim.models import FastText

class RentingRegressionDataset(Dataset):
    def __init__(self, dataframe, type_encoder, model, is_train=True):
        self.data = dataframe
        self.data.drop('_id', axis=1, inplace=True)
        self.data.drop('Deposit', axis=1, inplace=True)
        self.data.drop('Location', axis=1, inplace=True)
        self.data.drop('Cardinality', axis=1, inplace=True)
        self.embeddings_model = FastText.load("models/fastText_model")
        self.model = model

        # Apply label encoding to 'Floor' column
        self.data['Floor'] = self.data['Floor'].apply(self.extract_floor_number)

        # Convert specified boolean columns to integers
        self.boolean_columns = ['Storage room', 'Wardrobe', 'Furnished', 'Equipped kitchen', 'Renovation', 
                           'Reduced mobility', 'Heating', 'Garage', 'Elevator', 'Air conditioning',
                            'Swimming pool', 'Garden', 'Green areas', 'Terrace']
        self.data[self.boolean_columns] = self.data[self.boolean_columns].astype(int)

        # add floor and check the df head as it was causing problems
        self.numeric_columns = ['Built square meters', 'Plot square meters', 'Rooms', 'Bathrooms', 'Latitude', 'Longitude',
                                'Mean Price Location', 'Median Price Location', 'Std Price Location', 'Mean Price Type Location',
                                'Median Price Type Location', 'Std Price Type Location', 'Floor'] + self.boolean_columns
        if self.model == "beto":
            self._normalize_numeric_columns()
        
        # save dataset head to csv
        self.data.head().to_csv('data/dataset_head.csv')

        # One-hot encoding for 'Type' column
        self.type_encoder = type_encoder
        self.types_encoded = self.type_encoder.transform(self.data[['Type']])

        # Convert 'Description' to embeddings
        self.descriptions = self.convert_to_embeddings(self.data['Description'])

        # Combine all features
        self.features = np.hstack([self.types_encoded, self.descriptions, self.data.drop(columns=['Type', 'Description', 'Price'])])
        self.feature_names = \
            ["_".join(feature.split()) for feature in self.type_encoder.get_feature_names_out()] + \
            ['Description_' + str(i) for i in range(self.descriptions.shape[1])] + \
            list(self.data.drop(columns=['Type', 'Description', 'Price']).columns)

        self.targets = self.data['Price'].values

    def _normalize_numeric_columns(self):
        for column in self.numeric_columns:
            n_quantiles = min(1000, self.data[column].shape[0])
            self.data[column] = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal') \
                                    .fit_transform(self.data[[column]])

    def convert_to_embeddings(self, descriptions):
        return np.array([self.text_to_embeddings(desc) for desc in descriptions])
    
    def text_to_embeddings(self, text):
        """Converts a text description to a vector embedding by averaging the embeddings of the words in the description."""
        # Split the description into words and filter out empty words
        words = [word for word in text.split() if word]

        # Get embeddings for each word and average them
        word_embeddings = [self.embeddings_model.wv[word] for word in words if word in self.embeddings_model.wv]
        if word_embeddings:
            desc_embedding = np.mean(word_embeddings, axis=0)
        else:
            # Handle case with no words in model's vocabulary
            desc_embedding = np.zeros(self.embeddings_model.vector_size)
        
        return desc_embedding

    @staticmethod
    def extract_floor_number(floor):
        """
        Extracts the floor number from the 'Floor' column. There are several different values for this column,
        so we need to handle them all.

        You can look at the data to see the different values of the 'Floor' column:
        [x for x in df['Floor'].unique() if 'planta ' not in x or 'entreplanta' in x].

        Here are the values that are different from 'planta X' (where X is a number)
        and 'entreplanta', along with the values that we will assign to them:

            * 'bajo exterior', 'bajo interior', 'bajo': 0
            * 'semi-sótano interior', 'semi-sótano exterior': -1
            * 'sótano interior', 'sótano exterior': -2
        
        And the rest of the values are very differentiated to make them more "neutral":

            * 'entreplanta interior', 'entreplanta exterior': 2
            * 'exterior', 'interior': -999
            * 'no aplica': 999
        """
        if 'entreplanta' in floor:
            return 2
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
    def __init__(self, dataframe, batch_size=32, model="xgboost"):
        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size
        # - handle_unknown='ignore' tells the encoder to ignore any categories in the validation/test set that were not present in the training set
        # - sparse_output=False to return a numpy array instead of a sparse matrix
        self.type_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        self.possible_models = ["xgboost", "beto"]
        if model not in self.possible_models:
            raise ValueError(f"Model must be one of {self.possible_models}")
        self.model = model

    def setup(self, stage=None):
        # Split data into train, validation, and test sets
        train_val_df, test_df = train_test_split(self.dataframe, test_size=0.05) # (test: 5% of the original data)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2) # 0.2 x 0.95 = 0.19 (vaL: 19% of the original data)

        # Fit the OneHotEncoder here using only the training data
        self.type_encoder.fit(train_df[['Type']])

        self.train_dataset = RentingRegressionDataset(train_df, self.type_encoder, self.model, is_train=True)
        self.val_dataset = RentingRegressionDataset(val_df, self.type_encoder, self.model, is_train=False)
        self.test_dataset = RentingRegressionDataset(test_df, self.type_encoder, self.model, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('data/lat_long_preprocessed_data.csv')
    data_module = RegressionDataModule(df, batch_size=64, model="beto")
    data_module.setup()

    # for batch in data_module.train_dataloader():
    #     x, y = batch
    #     print(x.shape, y.shape)
    #     break
