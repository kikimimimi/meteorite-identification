import copy

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaInrEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding).squeeze(1)
        reconstruction = self.decoder(embedding)
        return logits, embedding, reconstruction


def standardize_fit(x, eps=1e-6):
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std


def standardize_apply(x, mean, std):
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def find_best_threshold(labels, probs):
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.95, 0.005):
        f1 = f1_score(labels, (probs > threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)


def make_loader(x, y, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_meta_inr_encoder(train_x, train_y, val_x, val_y, config, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mean, std = standardize_fit(train_x)
    train_x = standardize_apply(train_x, mean, std)
    val_x = standardize_apply(val_x, mean, std)

    hidden_dim = int(config.get("hidden_dim", 256))
    embedding_dim = int(config.get("embedding_dim", 64))
    dropout = float(config.get("dropout", 0.1))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    recon_weight = float(config.get("recon_weight", 0.05))
    epochs = int(config.get("epochs", 250))
    batch_size = int(config.get("batch_size", 64))
    patience = int(config.get("patience", 35))

    model = MetaInrEncoder(
        input_dim=train_x.shape[1],
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(DEVICE)

    positives = float(np.sum(train_y == 1))
    negatives = float(np.sum(train_y == 0))
    pos_weight_value = negatives / max(positives, 1.0)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=DEVICE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(train_x, train_y, batch_size, shuffle=True, seed=seed)
    val_tensor = torch.from_numpy(val_x.astype(np.float32)).to(DEVICE)
    val_y_np = val_y.astype(int)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_val_f1 = 0.0
    best_threshold = 0.5
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            logits, _, reconstruction = model(batch_x)
            loss = bce_loss(logits, batch_y) + recon_weight * mse_loss(reconstruction, batch_x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _, val_reconstruction = model(val_tensor)
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_recon = mse_loss(val_reconstruction, val_tensor).item()
        threshold, val_f1 = find_best_threshold(val_y_np, val_probs)
        score = val_f1 - 0.01 * val_recon
        if score > best_score:
            best_score = score
            best_val_f1 = val_f1
            best_threshold = threshold
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    print(
        f"Meta-INR encoder | best_epoch={best_epoch} | "
        f"val_f1={best_val_f1:.4f} | threshold={best_threshold:.3f}"
    )

    return {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "mean": mean,
        "std": std,
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "threshold": best_threshold,
        "config": dict(config),
    }


def transform_meta_inr_features(meta_checkpoint, inr_features, batch_size=512):
    mean = meta_checkpoint["mean"]
    std = meta_checkpoint["std"]
    x = standardize_apply(inr_features, mean, std)
    model = MetaInrEncoder(
        input_dim=int(meta_checkpoint["input_dim"]),
        hidden_dim=int(meta_checkpoint["hidden_dim"]),
        embedding_dim=int(meta_checkpoint["embedding_dim"]),
        dropout=float(meta_checkpoint.get("dropout", 0.0)),
    ).to(DEVICE)
    model.load_state_dict(meta_checkpoint["state_dict"])
    model.eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(DEVICE)
            logits, embedding, _ = model(batch)
            probs = torch.sigmoid(logits).unsqueeze(1)
            current = torch.cat([probs, embedding], dim=1).detach().cpu().numpy()
            outputs.append(current)
    return np.vstack(outputs).astype(np.float32)


def meta_feature_names(embedding_dim):
    return ["meta_inr_prob"] + [f"meta_inr_embedding_{idx}" for idx in range(int(embedding_dim))]
