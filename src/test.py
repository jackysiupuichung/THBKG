import argparse
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from src.models.graph_recbole import GraphRecBole
from src.models.temporal_gat import TemporalGAT

def main(config_file, checkpoint):
    config = Config(model='GraphRecBole', dataset='chembl', config_file_list=[config_file])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    num_users = dataset.num(config['USER_ID_FIELD'])
    num_items = dataset.num(config['ITEM_ID_FIELD'])
    graph_model = TemporalGAT(num_users, num_items, hidden_dim=128)

    model = GraphRecBole(config, dataset, graph_model)

    trainer = Trainer(config, model)
    test_result = trainer.evaluate(test_data, load_best_model=True, model_file=checkpoint)

    print("✅ Test Results:")
    print(test_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/recbole.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/graph_recbole.pth")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
