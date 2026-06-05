import os
import argparse
import torch
import pandas as pd
import numpy as np

from neumf import NeuMFEngine
from data_utils import SampleGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal NeuMF with visual features")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/microlens-5k",
        help="Path to dataset directory"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )

    parser.add_argument(
        "--num_epoch",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size"
    )

    parser.add_argument(
        "--use_cuda",
        action="store_true",
        default=False,
        help="Use CUDA if available"
    )

    return parser.parse_args()


# =========================================================
# Config chỉ dùng cho Multimodal NeuMF
# =========================================================
neumf_config = {
    "alias": "neumf_factor8neg4",

    "num_epoch": 20,
    "batch_size": 1024,

    "optimizer": "adam",
    "adam_lr": 1e-3,

    "num_users": 6040,
    "num_items": 3706,

    "latent_dim_mf": 8,
    "latent_dim_mlp": 8,

    "num_negative": 4,

    "layers": [16, 64, 32, 16, 8],

    "l2_regularization": 0.0000001,
    "weight_init_gaussian": True,

    "use_cuda": False,
    "use_bachify_eval": True,
    "device_id": 0,

    "pretrain": False,
    "pretrain_mf": "checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model",
    "pretrain_mlp": "checkpoints/mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model",

    # sẽ được cập nhật tự động sau khi load visual_embeddings.pt
    "visual_dim": 768,

    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}


def load_visual_embeddings(data_dir, item_id_map):
    """
    Load visual_embeddings.pt và map item gốc -> itemId mới.

    Hỗ trợ 2 dạng:
    1. dict: {original_item_id: embedding}
    2. tensor: shape [num_items, visual_dim]
    """

    visual_path = os.path.join(data_dir, "visual_embeddings.pt")

    if not os.path.exists(visual_path):
        raise FileNotFoundError(f"Không tìm thấy visual_embeddings.pt tại: {visual_path}")

    raw_visual = torch.load(visual_path, weights_only=False)

    visual_embeddings = {}

    # Mapping item gốc -> itemId mới
    orig_to_new = {
        str(orig_item): int(new_item_id)
        for orig_item, new_item_id in zip(item_id_map["item"], item_id_map["itemId"])
    }

    # Trường hợp 1: visual_embeddings.pt là dict
    if isinstance(raw_visual, dict):
        for original_item, emb in raw_visual.items():
            key = str(original_item)

            if key in orig_to_new:
                new_item_id = orig_to_new[key]

                if not torch.is_tensor(emb):
                    emb = torch.tensor(emb, dtype=torch.float32)

                visual_embeddings[new_item_id] = emb.float()

    # Trường hợp 2: visual_embeddings.pt là tensor
    elif torch.is_tensor(raw_visual):
        raw_visual = raw_visual.float()

        item_id_map_sorted = item_id_map.sort_values("itemId").reset_index(drop=True)

        for row in item_id_map_sorted.itertuples():
            original_item = int(row.item)
            new_item_id = int(row.itemId)

            # MicroLens thường đánh item gốc từ 1, tensor index từ 0
            visual_embeddings[new_item_id] = raw_visual[original_item - 1]

    else:
        raise TypeError("visual_embeddings.pt phải là dict hoặc torch.Tensor")

    if len(visual_embeddings) == 0:
        raise ValueError(
            "Không map được visual embeddings. "
            "Kiểm tra kiểu item trong pairs.csv và key trong visual_embeddings.pt."
        )

    sample_v = next(iter(visual_embeddings.values()))
    visual_dim = sample_v.shape[-1]

    print(f"Visual embeddings: {len(visual_embeddings)} items")
    print(f"Visual dim: {visual_dim}")

    return visual_embeddings, visual_dim


def main():
    args = parse_args()

    # =========================================================
    # 1. Load interactions
    # =========================================================
    pairs_path = os.path.join(args.data_dir, "pairs.csv")

    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Không tìm thấy pairs.csv tại: {pairs_path}")

    interactions = pd.read_csv(pairs_path)

    # MicroLens là implicit feedback
    interactions["rating"] = 1.0

    # =========================================================
    # 2. Reindex user
    # =========================================================
    user_id_map = interactions[["user"]].drop_duplicates().reset_index(drop=True)
    user_id_map["userId"] = np.arange(len(user_id_map))

    interactions = pd.merge(
        interactions,
        user_id_map,
        on="user",
        how="left"
    )

    # =========================================================
    # 3. Reindex item
    # =========================================================
    item_id_map = interactions[["item"]].drop_duplicates().reset_index(drop=True)
    item_id_map["itemId"] = np.arange(len(item_id_map))

    interactions = pd.merge(
        interactions,
        item_id_map,
        on="item",
        how="left"
    )

    ratings = interactions[["userId", "itemId", "rating", "timestamp"]]

    print("Range of userId is [{}, {}]".format(
        ratings.userId.min(),
        ratings.userId.max()
    ))

    print("Range of itemId is [{}, {}]".format(
        ratings.itemId.min(),
        ratings.itemId.max()
    ))

    print("Number of users:", ratings["userId"].nunique())
    print("Number of items:", ratings["itemId"].nunique())
    print("Number of interactions:", len(ratings))

    # =========================================================
    # 4. Load visual embeddings
    # =========================================================
    visual_embeddings, visual_dim = load_visual_embeddings(
        args.data_dir,
        item_id_map
    )

    # =========================================================
    # 5. Config cho NeuMF
    # =========================================================
    config = neumf_config.copy()

    config["num_users"] = ratings["userId"].nunique()
    config["num_items"] = ratings["itemId"].nunique()

    config["num_epoch"] = args.num_epoch
    config["batch_size"] = args.batch_size
    config["use_cuda"] = args.use_cuda

    config["visual_dim"] = visual_dim

    config["model_dir"] = os.path.join(
        args.checkpoint_dir,
        "{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model"
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Config:")
    for k, v in config.items():
        print(f"{k}: {v}")

    # =========================================================
    # 6. SampleGenerator
    # =========================================================
    sample_generator = SampleGenerator(
        ratings=ratings,
        visual_embeddings=visual_embeddings,
        maxlen=50
    )

    evaluate_data = sample_generator.evaluate_data

    # =========================================================
    # 7. Train Multimodal NeuMF
    # =========================================================
    engine = NeuMFEngine(config)

    for epoch in range(config["num_epoch"]):
        print("\nEpoch {} starts!".format(epoch))
        print("-" * 80)

        train_loader = sample_generator.instance_a_train_loader(
            config["num_negative"],
            config["batch_size"]
        )

        engine.train_an_epoch(
            train_loader,
            epoch_id=epoch
        )

        hit_ratio, ndcg = engine.evaluate(
            evaluate_data,
            epoch_id=epoch
        )

        engine.save(
            config["alias"],
            epoch,
            hit_ratio,
            ndcg
        )


if __name__ == "__main__":
    main()