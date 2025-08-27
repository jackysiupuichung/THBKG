import optuna
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from src.models.graph_recbole import GraphRecBole
from src.models.temporal_gat import TemporalGAT
import yaml

def objective(trial, config_path="configs/recbole.yaml"):
    # Load base config
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Sample hyperparameters from yaml ranges
    hidden_dim = trial.suggest_int("hidden_dim", 
                                   cfg_dict["graph_model"]["hidden_dim"]["range"][0],
                                   cfg_dict["graph_model"]["hidden_dim"]["range"][1])
    dropout = trial.suggest_float("dropout", 
                                  cfg_dict["graph_model"]["dropout"]["range"][0],
                                  cfg_dict["graph_model"]["dropout"]["range"][1])
    gat_heads = trial.suggest_categorical("gat_heads", cfg_dict["graph_model"]["gat_heads"]["choices"])

    # Override defaults
    cfg_dict["graph_model"]["hidden_dim"]["default"] = hidden_dim
    cfg_dict["graph_model"]["dropout"]["default"] = dropout
    cfg_dict["graph_model"]["gat_heads"]["default"] = gat_heads

    # Build RecBole config
    config = Config(model='GraphRecBole', dataset=cfg_dict["dataset"], config_dict=cfg_dict)

    # Dataset
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Graph model
    num_users = dataset.num(config["USER_ID_FIELD"])
    num_items = dataset.num(config["ITEM_ID_FIELD"])
    graph_model = TemporalGAT(num_users, num_items, hidden_dim=hidden_dim)

    # RecBole model wrapper
    model = GraphRecBole(config, dataset, graph_model)

    # Trainer
    trainer = Trainer(config, model)
    best_valid_score, _ = trainer.fit(train_data, valid_data)

    # Objective: maximize Recall@10
    return best_valid_score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_trial.params)
