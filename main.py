import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from models.lstm_attention import LSTMAttention
from models.rnn_cell import RNNCell


def get_model(name, config):
    """Factory function to initialize models."""
    if name.lower() == "rnn":
        return RNNCell(config.input_size, config.hidden_size, config.output_size)
    elif name.lower() == "lstm_attention":
        return LSTMAttention(
            input_dim=config.input_size,
            hidden_dim=config.hidden_size,
            output_dim=config.output_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
        )
    else:
        raise ValueError(f"Unknown model type: {name}")


def train(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs, _ = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")



if __name__ == "__main__":
    from config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm_attention")
    parser.add_argument("--data_path", type=str, default="./data/train.csv")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloader(args.data_path, cfg.batch_size)

    model = get_model(args.model, cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Training {args.model.upper()} on {args.data_path}...")
    train(model, train_loader, criterion, optimizer, device, args.epochs)
