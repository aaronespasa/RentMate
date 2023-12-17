import torch
from torch import nn
from transformers import BertForSequenceClassification, AutoModel, AutoConfig, logging

from models.mlp import MLP # change this to from mlp import MLP if you're running this file directly
# from mlp import MLP

# disable verbosity warnings from transformers
logging.set_verbosity_error()

class MultimodalTransformerRegressor(BertForSequenceClassification):
    """
    A model for regression which combines text, categorical and numberical features.
    The text features are processed with a BERT model, and the categorical and numerical
    features are processed with an MLP for final regression.
    """
    def __init__(self, cat_feat_dim, numerical_feat_dim, num_labels=1):
        custom_bert_config = AutoConfig.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
        custom_bert_config.num_labels = num_labels

        # Initialize the parent class with the custom config
        super().__init__(custom_bert_config)

        # Override the parent class's bert model with the custom one
        self.bert = AutoModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', output_loading_info = False)

        self.num_labels = num_labels # 1 as it's a regression task

        # Calculate the combined input dimension for the MLP
        self.mlp_input_dim = self.bert.config.hidden_size + cat_feat_dim + numerical_feat_dim

        # Create a batch normalizer for the numerical features
        self.numeric_batch_norm = nn.BatchNorm1d(numerical_feat_dim)

        # We're following the MultiModal-Toolkit (https://github.com/georgian-io/Multimodal-Toolkit)
        # where each layer of the MLP has 1/4th the number of neurons as the previous layer
        dims = []
        dim = self.mlp_input_dim
        while dim > 1:
            dim = dim // 4
            if dim > 1:
                dims.append(dim)
        
        # Print out the resulting MLP dimensions
        print("MLP layer sizes:",
              f" * Input: {self.mlp_input_dim}",
              f" * Hidden: {dims}",
              f" * Output: {self.num_labels}", sep="\n", end="\n\n")

        self.mlp = MLP(input_dim=self.mlp_input_dim, output_dim=self.num_labels, num_hidden_layers=len(dims), hidden_dimensions=dims)

    def forward(self, input_ids, attention_mask, categorical_features, numerical_features):
        ###########################################
        # 1. Run the BERT model on the text input #
        ###########################################
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # outputs[0] - All of the output embeddings from BERT
        # outputs[1] - The [CLS] token embedding, with some additional "pooling" applied
        cls_output = outputs[1]

        # Apply dropout to the [CLS] token embedding
        cls_output = self.dropout(cls_output)

        ############################################
        # 2. Concatenate all of the input features #
        ############################################

        # Apply batch normalization to the numerical features
        numerical_features = self.numeric_batch_norm(numerical_features)

        # Concatenate all of the features
        combined_features = torch.cat([cls_output, categorical_features, numerical_features], dim=1)

        ###########################################
        # 3. Run the MLP on the combined features #
        ###########################################
        output = self.mlp(combined_features).squeeze(-1)

        return output

if __name__ == "__main__":
    import os
    import sys
    # Example usage
    import pandas as pd
    from torchmetrics.regression.r2 import R2Score
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dataset import RegressionDataModule
    from utils.loss.rmse import RMSELoss
    
    df = pd.read_csv('data/lat_long_preprocessed_data.csv')
    model = "beto"
    data_module = RegressionDataModule(df, batch_size=64, model=model)
    data_module.setup()

    numerical_features, categorical_features, input_ids, attention_mask, target = None, None, None, None, None

    for batch in data_module.train_dataloader():
        numerical_features, categorical_features, input_ids, attention_mask, target = batch
        break

    model = MultimodalTransformerRegressor(cat_feat_dim=categorical_features.shape[1],
                                           numerical_feat_dim=numerical_features.shape[1])
    
    output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   categorical_features=categorical_features,
                   numerical_features=numerical_features)
    
    mseloss = nn.MSELoss()(output, target).item()
    maeloss = nn.L1Loss()(output, target).item()
    rmse = RMSELoss()(output, target).item()
    r2 = R2Score()(output, target).item()

    # print the output beautifully
    print("MSE Loss:", mseloss)
    print("MAE Loss:", maeloss)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("Logits shape:", output.shape)

