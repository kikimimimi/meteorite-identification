import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def pil_resize_resample():
    try:
        return Image.Resampling.BICUBIC
    except AttributeError:
        return Image.BICUBIC


def stable_inr_cache_name(prefix, paths, config):
    digest = hashlib.md5()
    digest.update(str(config).encode("utf-8"))
    for path in paths:
        digest.update(str(path).encode("utf-8"))
    return f"{prefix}_inr_{digest.hexdigest()[:12]}.npz"


def normalize_rows(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class TinySiren(nn.Module):
    def __init__(self, hidden_dim=32, hidden_layers=2, omega_0=30.0):
        super().__init__()
        layers = [SineLayer(2, hidden_dim, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.ModuleList(layers)
        self.final = nn.Linear(hidden_dim, 3)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / omega_0
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.uniform_(-bound, bound)

    def forward(self, coords, return_hidden=False):
        hidden = coords
        for layer in self.net:
            hidden = layer(hidden)
        rgb = torch.sigmoid(self.final(hidden))
        if return_hidden:
            return rgb, hidden
        return rgb


def make_model(hidden_dim, hidden_layers, omega_0, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TinySiren(hidden_dim=hidden_dim, hidden_layers=hidden_layers, omega_0=omega_0)
    return model.to(DEVICE)


def make_coord_grid(image_size):
    values = torch.linspace(-1.0, 1.0, image_size, device=DEVICE)
    try:
        yy, xx = torch.meshgrid(values, values, indexing="ij")
    except TypeError:
        yy, xx = torch.meshgrid(values, values)
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def load_image_tensor(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), pil_resize_resample())
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array.reshape(-1, 3)).to(DEVICE)


def flatten_parameters(model):
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()]).numpy()


def describe_vector(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    return [
        float(values.mean()),
        float(values.std()),
        float(np.mean(np.abs(values))),
        float(np.sqrt(np.mean(values * values))),
        float(np.max(np.abs(values))),
    ], [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_abs_mean",
        f"{prefix}_rms",
        f"{prefix}_max_abs",
    ]


def describe_channels(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    stats = []
    names = []
    for channel in range(values.shape[1]):
        channel_values = values[:, channel]
        channel_name = f"{prefix}_c{channel}"
        current, current_names = describe_vector(channel_values, channel_name)
        stats.extend(current)
        names.extend(current_names)
    return stats, names


def fit_single_inr_descriptor(path, config, image_index=0):
    image_size = int(config["image_size"])
    steps = int(config["steps"])
    pixels_per_step = int(config["pixels_per_step"])
    hidden_dim = int(config["hidden_dim"])
    hidden_layers = int(config["hidden_layers"])
    omega_0 = float(config["omega_0"])
    lr = float(config["lr"])
    seed = int(config["seed"])

    coords = make_coord_grid(image_size)
    target = load_image_tensor(path, image_size)
    model = make_model(hidden_dim, hidden_layers, omega_0, seed)
    torch.manual_seed(seed + image_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + image_index)
    initial_params = flatten_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with torch.no_grad():
        initial_pred = model(coords)
        initial_loss = torch.mean((initial_pred - target) ** 2).item()

    losses = []
    num_pixels = coords.shape[0]
    for step in range(steps):
        if pixels_per_step > 0 and pixels_per_step < num_pixels:
            pixel_idx = torch.randint(0, num_pixels, (pixels_per_step,), device=DEVICE)
            step_coords = coords[pixel_idx]
            step_target = target[pixel_idx]
        else:
            step_coords = coords
            step_target = target

        pred = model(step_coords)
        loss = torch.mean((pred - step_target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        final_pred, hidden = model(coords, return_hidden=True)
        residual = final_pred - target
        final_loss = torch.mean(residual ** 2).item()

    final_params = flatten_parameters(model)
    delta = final_params - initial_params
    hidden_np = hidden.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    pred_np = final_pred.detach().cpu().numpy()
    residual_np = residual.detach().cpu().numpy()

    descriptor = []
    names = []
    loss_curve = np.asarray(losses, dtype=np.float32)
    checkpoints = [0, max(0, steps // 4 - 1), max(0, steps // 2 - 1), max(0, (3 * steps) // 4 - 1), steps - 1]
    for idx, checkpoint in enumerate(checkpoints):
        descriptor.append(float(loss_curve[checkpoint]))
        names.append(f"loss_step_{idx}")

    scalar_values = [
        float(initial_loss),
        float(final_loss),
        float(initial_loss - final_loss),
        float(-10.0 * np.log10(max(final_loss, 1e-8))),
    ]
    scalar_names = ["initial_mse", "final_mse", "mse_drop", "psnr"]
    descriptor.extend(scalar_values)
    names.extend(scalar_names)

    for values, prefix in [
        (delta, "param_delta"),
        (final_params, "param_final"),
    ]:
        current, current_names = describe_vector(values, prefix)
        descriptor.extend(current)
        names.extend(current_names)

    for values, prefix in [
        (target_np, "image_rgb"),
        (pred_np, "recon_rgb"),
        (residual_np, "residual_rgb"),
        (np.abs(residual_np), "abs_residual_rgb"),
    ]:
        current, current_names = describe_channels(values, prefix)
        descriptor.extend(current)
        names.extend(current_names)

    hidden_mean = hidden_np.mean(axis=0)
    hidden_std = hidden_np.std(axis=0)
    hidden_abs_mean = np.abs(hidden_np).mean(axis=0)
    descriptor.extend(hidden_mean.tolist())
    names.extend([f"hidden_mean_{idx}" for idx in range(hidden_dim)])
    descriptor.extend(hidden_std.tolist())
    names.extend([f"hidden_std_{idx}" for idx in range(hidden_dim)])
    descriptor.extend(hidden_abs_mean.tolist())
    names.extend([f"hidden_abs_mean_{idx}" for idx in range(hidden_dim)])

    return np.asarray(descriptor, dtype=np.float32), names


def extract_inr_features(paths, cache_dir, prefix="inr", **kwargs):
    paths = [Path(path) for path in paths]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "image_size": int(kwargs.get("image_size", 48)),
        "steps": int(kwargs.get("steps", 80)),
        "pixels_per_step": int(kwargs.get("pixels_per_step", 1024)),
        "hidden_dim": int(kwargs.get("hidden_dim", 32)),
        "hidden_layers": int(kwargs.get("hidden_layers", 2)),
        "omega_0": float(kwargs.get("omega_0", 30.0)),
        "lr": float(kwargs.get("lr", 1e-3)),
        "seed": int(kwargs.get("seed", 123)),
    }
    cache_path = cache_dir / stable_inr_cache_name(prefix, paths, config)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["feature_names"].tolist()

    features = []
    feature_names = None
    for index, path in enumerate(tqdm(paths, desc=f"Extract {prefix} INR")):
        descriptor, names = fit_single_inr_descriptor(path, config, image_index=index)
        features.append(descriptor)
        feature_names = names

    features = np.vstack(features).astype(np.float32)
    np.savez_compressed(
        cache_path,
        features=features,
        feature_names=np.array(feature_names, dtype=object),
        paths=np.array([str(path) for path in paths], dtype=object),
        config=np.array([config], dtype=object),
    )
    print(f"Cached INR features: {cache_path}")
    return features, feature_names


def list_image_paths(root):
    root = Path(root)
    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--cache-dir", default="features")
    parser.add_argument("--prefix", default="inr")
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--pixels-per-step", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--omega-0", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    paths = list_image_paths(args.image_root)
    features, _ = extract_inr_features(
        paths,
        args.cache_dir,
        prefix=args.prefix,
        image_size=args.image_size,
        steps=args.steps,
        pixels_per_step=args.pixels_per_step,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        omega_0=args.omega_0,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"Extracted INR features: samples={features.shape[0]} dim={features.shape[1]}")


if __name__ == "__main__":
    main()
