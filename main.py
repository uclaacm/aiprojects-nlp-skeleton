import os

import constants
from data.StartingDataset import StartingDataset
from data.embedWrapping import GoogleNewsEmbeddor
from data.EmbeddingDataset import EmbeddingsDataset
from networks.StartingNetwork import StartingNetwork
from networks.VariableLSTM import VariableLSTM
from train_functions.starting_train import starting_train
import torch

import argparse
import configparser

embeddor_type_matching = {
    "google_news": GoogleNewsEmbeddor
}


def generate_embedding(data_path, dataset_config):
    embeddor_type = dataset_config["embedding_type"]
    embeddor = embeddor_type_matching[embeddor_type](dataset_config["embeddor_path"], DEBUG=True)
    return EmbeddingsDataset(data_path, embeddor)


model_type_matching = {
    "starting": StartingNetwork,
    "lstm": VariableLSTM
}

dataset_type_match = {
    "starting": StartingDataset,
    "embedding": generate_embedding
}


def main(dataset_type, model_type, dataset_config, model_config, data_path, train_hyperparameters, random_seed=42):
    # Get command line arguments
    hyperparameters = {"epochs": train_hyperparameters["epochs"], "batch_size": train_hyperparameters["batch_size"]}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!

    dataset = dataset_type_match[dataset_type](data_path, dataset_config)
    # splits dataset into two with random indices
    trainSize = int(train_hyperparameters["train_split"] * len(dataset))
    valSize = len(dataset) - trainSize
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [trainSize, valSize],
        generator=torch.Generator().manual_seed(random_seed))
    model = model_type_matching[model_type](model_config)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=train_hyperparameters["n_eval"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path", type=str, default="./config/starting.ini")
    parser.add_argument("--data_path", type=str, default="train.csv")

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    train_hyperparameters = {
        "epochs": int(config["TRAINING"]["EPOCHS"]), 
        "batch_size": int(config["TRAINING"]["BATCH_SIZE"]),
        "n_eval": int(config["TRAINING"]["N_EVAL"]),
        "train_split": float(config["DATASET"]["SPLIT"])
    }

    main(
        config["DATASET"]["TYPE"], 
        config["MODEL"]["TYPE"], 
        config["DATASET"],
        config["MODEL"], 
        args.data_path, 
        train_hyperparameters
        )
