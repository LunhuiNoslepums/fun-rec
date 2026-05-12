"""
离线评估脚本：加载已训练的模型和测试数据，计算召回和精排指标

使用方式:
    source .venv/bin/activate
    python backend/offline/evaluation/run_evaluation.py

依赖:
    - tensorflow==2.15.0
    - numpy pandas scikit-learn tqdm
    - funrec 库的 layers.py (通过 importlib 直接加载，避免完整依赖链)
"""

import os
import sys
import pickle
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

# ── 路径配置 ──────────────────────────────────────────────────────────
PROCESSED_DIR = Path("/Users/huilun/Desktop/Code/rec/fun_processed_data/web_project")
SAVED_MODELS_DIR = PROCESSED_DIR / "saved_models"
SRC_DIR = Path("/Users/huilun/Desktop/Code/rec/fun-rec/src")

# ── 直接加载 funrec.layers (避免触发 funrec/__init__.py 的完整依赖链) ──
_spec = importlib.util.spec_from_file_location(
    "layers", str(SRC_DIR / "funrec/models/layers.py")
)
_layers_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layers_mod)
DNNs = _layers_mod.DNNs
FM = _layers_mod.FM


# ══════════════════════════════════════════════════════════════════════
# 第一部分：召回评估 (YouTubeDNN)
# ══════════════════════════════════════════════════════════════════════

def evaluate_retrieval():
    print("=" * 60)
    print("召回模型评估 (YouTubeDNN)")
    print("=" * 60)

    # 1. 加载测试数据
    print("\n[1/5] 加载召回测试数据...")
    data = pickle.load(open(str(PROCESSED_DIR / "train_eval_sample_final.pkl"), "rb"))
    test = data["test"]
    feature_dict = pickle.load(open(str(PROCESSED_DIR / "feature_dict.pkl"), "rb"))
    vocab_size = feature_dict["movie_id"]
    print(f"    测试样本数: {len(test['user_id'])}")
    print(f"    物品词表大小: {vocab_size}")

    # 2. 加载模型
    print("\n[2/5] 加载召回模型...")
    user_model = tf.keras.models.load_model(str(SAVED_MODELS_DIR / "user_model"))
    item_model = tf.keras.models.load_model(str(SAVED_MODELS_DIR / "item_model"))
    print("    用户模型和物品模型加载完成")

    # 3. 计算用户 embeddings
    print("\n[3/5] 计算用户 embeddings...")
    user_inputs = {
        "user_id": test["user_id"].reshape(-1, 1),
        "gender": test["gender"].reshape(-1, 1),
        "age": test["age"].reshape(-1, 1),
        "occupation": test["occupation"].reshape(-1, 1),
        "zip_code": test["zip_code"].reshape(-1, 1),
        "hist_movie_id": test["hist_movie_id"].reshape(-1, 10),
        "hist_genres": test["hist_genres"].reshape(-1, 10),
    }
    user_embs = user_model.predict(user_inputs, batch_size=256, verbose=0)
    print(f"    用户 embeddings: {user_embs.shape}")

    # 4. 计算物品 embeddings (所有物品)
    print("\n[4/5] 计算物品 embeddings...")
    all_item_ids = np.arange(vocab_size, dtype=np.int32).reshape(-1, 1)
    item_embs = item_model.predict({"movie_id": all_item_ids}, batch_size=256, verbose=0)
    print(f"    物品 embeddings: {item_embs.shape}")

    # 5. 评估 HitRate@K, NDCG@K, Precision@K
    print("\n[5/5] 计算召回指标...")

    # 物品 ID 范围: 1..vocab_size-1 (0 是填充/未知，不作为推荐候选)
    valid_item_ids = np.arange(1, vocab_size, dtype=np.int32)
    item_embs = item_embs[valid_item_ids]  # 排除 padding (id=0)

    test_users = test["user_id"]
    test_items = test["movie_id"]
    unique_users = np.unique(test_users)
    print(f"    评估用户数: {len(unique_users)}")
    print(f"    候选物品数: {len(valid_item_ids)}")

    n_items = len(valid_item_ids)
    k_list = [5, 10, 20]
    metrics = {f"hit_rate@{k}": [] for k in k_list}
    metrics.update({f"ndcg@{k}": [] for k in k_list})
    metrics.update({f"precision@{k}": [] for k in k_list})

    for user_id in unique_users:
        mask = test_users == user_id
        relevant = set(test_items[mask])
        if not relevant:
            continue

        user_idx = np.where(mask)[0][0]
        user_emb = user_embs[user_idx : user_idx + 1]

        scores = cosine_similarity(user_emb, item_embs)[0]
        # topk_indices 是 valid_item_ids 中的索引，实际 movie_id = valid_item_ids[topk_indices]
        topk_indices = valid_item_ids[np.argsort(-scores)]

        for k in k_list:
            recommended = list(topk_indices[:k])
            hit = 1.0 if len(set(recommended) & relevant) > 0 else 0.0
            metrics[f"hit_rate@{k}"].append(hit)

            precision = len(set(recommended) & relevant) / k
            metrics[f"precision@{k}"].append(precision)

            dcg = sum(
                1.0 / np.log2(pos + 2)
                for pos, item in enumerate(recommended)
                if item in relevant
            )
            idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f"ndcg@{k}"].append(ndcg)

    # 汇总 + 随机基线对比
    print("\n" + "-" * 60)
    print("召回评估结果 (YouTubeDNN):")
    print("-" * 60)
    random_hr = {k: 1 - ((n_items - k) / n_items) ** 1 for k in k_list}  # 近似
    for k in k_list:
        hr = np.mean(metrics[f"hit_rate@{k}"])
        nd = np.mean(metrics[f"ndcg@{k}"])
        pr = np.mean(metrics[f"precision@{k}"])
        rand_hr = k / n_items
        print(f"  HitRate@{k}:    {hr:.4f}  (随机基线: {rand_hr:.4f})")
        print(f"  NDCG@{k}:       {nd:.4f}")
        print(f"  Precision@{k}:  {pr:.4f}")
        print()

    return metrics


# ══════════════════════════════════════════════════════════════════════
# 第二部分：精排评估 (DeepFM)
# ══════════════════════════════════════════════════════════════════════

def group_auc(labels, preds, user_ids):
    """计算 gAUC (group AUC)：每个用户单独算 AUC 后按样本数加权平均"""
    df = pd.DataFrame({"user_id": user_ids, "label": labels, "pred": preds})
    total_weight = 0.0
    weighted_auc = 0.0
    for _, group in df.groupby("user_id"):
        if len(group) < 2:
            continue
        ulabel = group["label"].values
        upred = group["pred"].values
        # 组内必须有正负样本才能算 AUC
        if ulabel.sum() == 0 or ulabel.sum() == len(ulabel):
            continue
        try:
            auc = roc_auc_score(ulabel, upred)
        except ValueError:
            continue
        weight = len(group)
        weighted_auc += auc * weight
        total_weight += weight
    return weighted_auc / total_weight if total_weight > 0 else 0.0


def evaluate_ranking():
    print("\n" + "=" * 60)
    print("精排模型评估 (DeepFM)")
    print("=" * 60)

    # 1. 加载测试数据
    print("\n[1/4] 加载精排测试数据...")
    data = pickle.load(open(str(PROCESSED_DIR / "ranking_train_eval_sample.pkl"), "rb"))
    test = data["test"]
    print(f"    测试样本数: {len(test['is_click'])}")
    pos_ratio = test["is_click"].mean()
    print(f"    正样本比例: {pos_ratio:.2%}")

    # 2. 加载模型
    print("\n[2/4] 加载精排模型...")
    model = tf.keras.models.load_model(
        str(SAVED_MODELS_DIR / "ranking_model"),
        custom_objects={"DNNs": DNNs, "FM": FM},
    )

    # 3. 预测 CTR
    print("\n[3/4] CTR 预测...")
    feature_cols = ["user_id", "gender", "age", "occupation", "zip_code",
                    "movie_id", "genres", "isAdult", "startYear"]
    model_input = {col: test[col].reshape(-1, 1) for col in feature_cols}
    preds = model.predict(model_input, batch_size=1024, verbose=0).flatten()
    print(f"    预测完成: {len(preds)} 个样本")

    # 4. 计算 AUC 和 gAUC
    print("\n[4/4] 计算指标...")
    labels = test["is_click"].flatten()
    auc = roc_auc_score(labels, preds)
    gauc = group_auc(labels, preds, test["user_id_original"])

    print("\n" + "-" * 40)
    print("精排评估结果:")
    print("-" * 40)
    print(f"  AUC:  {auc:.4f}")
    print(f"  gAUC: {gauc:.4f}")
    print()

    return {"auc": auc, "gauc": gauc}


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 关闭 TF 非关键日志

    retrieval_metrics = evaluate_retrieval()
    ranking_metrics = evaluate_ranking()

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    print("\n召回指标:")
    for k in [5, 10, 20]:
        print(f"  HitRate@{k} = {np.mean(retrieval_metrics[f'hit_rate@{k}']):.4f}"
              f"   NDCG@{k} = {np.mean(retrieval_metrics[f'ndcg@{k}']):.4f}"
              f"   Precision@{k} = {np.mean(retrieval_metrics[f'precision@{k}']):.4f}")
    print(f"\n精排指标:")
    print(f"  AUC  = {ranking_metrics['auc']:.4f}")
    print(f"  gAUC = {ranking_metrics['gauc']:.4f}")
