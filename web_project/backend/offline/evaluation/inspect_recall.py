"""
YouTubeDNN 召回效果检查脚本

随机抽取几个测试用户，看双塔模型召回了什么电影，
与实际交互过的电影对比，验证召回质量。
"""

import os
import pickle
import random
import numpy as np
import tensorflow as tf

PROCESSED_DIR = "/Users/huilun/Desktop/Code/rec/fun_processed_data/web_project"

print("=" * 60)
print("YouTubeDNN 召回效果抽查")
print("=" * 60)

# 1. 加载数据
print("\n[1/4] 加载数据...")
data = pickle.load(open(f"{PROCESSED_DIR}/train_eval_sample_final.pkl", "rb"))
test = data["test"]
train = data["train"]

# 整理每个用户的交互历史（从训练集）
user_history = {}
for i in range(len(train["user_id"])):
    uid = train["user_id"][i]
    mid = train["movie_id"][i]
    if uid not in user_history:
        user_history[uid] = set()
    user_history[uid].add(mid)

# 整理每个用户的测试样本 ground truth
user_ground_truth = {}
for i in range(len(test["user_id"])):
    uid = test["user_id"][i]
    mid = test["movie_id"][i]
    if uid not in user_ground_truth:
        user_ground_truth[uid] = []
    user_ground_truth[uid].append(mid)

print(f"  训练集用户数: {len(user_history)}")
print(f"  测试集用户数: {len(user_ground_truth)}")

# 2. 加载模型
print("\n[2/4] 加载 YouTubeDNN 模型...")
user_model = tf.keras.models.load_model(f"{PROCESSED_DIR}/saved_models/user_model")
item_embeddings = np.load(f"{PROCESSED_DIR}/saved_models/item_embeddings.npy")

# 构建带 padding 的 embedding 矩阵（与 recall 逻辑一致）
dim = item_embeddings.shape[1]
item_embedding_matrix = np.zeros((item_embeddings.shape[0] + 1, dim), dtype=np.float32)
item_embedding_matrix[1:] = item_embeddings

print(f"  模型输出维度: {user_model.output.shape}")
print(f"  物品总数: {item_embedding_matrix.shape[0] - 1} (不含 padding)")

# 加载原始电影 ID 映射
vocab_dict = pickle.load(open(f"{PROCESSED_DIR}/vocab_dict.pkl", "rb"))
all_movie_ids = list(vocab_dict["movie_id"])

# 加载电影信息供展示
import pandas as pd
df_movies = pd.read_pickle("/Users/huilun/Desktop/Code/rec/funrec-movielens-1m/movies.pkl")
movie_id_to_title = dict(zip(df_movies["movie_id"], df_movies.get("title", df_movies["movie_id"])))
movie_id_to_genres = dict(zip(df_movies["movie_id"], df_movies["genres"]))

# 3. 抽几个用户做召回检查
print("\n[3/4] 随机抽取用户进行抽查...")

# 选一些交互历史丰富的用户
user_ids = sorted(user_ground_truth.keys())
random.seed(42)
sample_users = random.sample(user_ids, min(5, len(user_ids)))

for uid in sample_users:
    print(f"\n{'─' * 50}")

    # 获取测试样本的用户特征
    mask = test["user_id"] == uid
    idx = np.where(mask)[0][0]

    # 构建模型输入
    model_inputs = {
        "user_id": test["user_id"][idx].reshape(1, 1),
        "gender": test["gender"][idx].reshape(1, 1),
        "age": test["age"][idx].reshape(1, 1),
        "occupation": test["occupation"][idx].reshape(1, 1),
        "zip_code": test["zip_code"][idx].reshape(1, 1),
        "hist_movie_id": test["hist_movie_id"][idx].reshape(1, 10),
        "hist_genres": test["hist_genres"][idx].reshape(1, 10),
    }

    # 预测
    user_emb = user_model.predict(model_inputs, verbose=0)
    scores = np.dot(user_emb, item_embedding_matrix.T)[0]

    # Top-10 召回
    top_indices = np.argsort(scores)[::-1][:20]

    # Ground truth 电影
    gt_movie_ids_raw = user_ground_truth[uid]
    # 解码为原始 ID
    gt_raw = []
    for enc_mid in gt_movie_ids_raw:
        if enc_mid - 1 < len(all_movie_ids):
            gt_raw.append(all_movie_ids[enc_mid - 1])

    # 用户历史电影
    hist_raw = set()
    for enc_mid in user_history.get(uid, set()):
        if enc_mid - 1 < len(all_movie_ids):
            hist_raw.add(all_movie_ids[enc_mid - 1])

    print(f"用户 ID: {uid}")
    print(f"  交互历史: {len(hist_raw)} 部电影")
    print(f"  Ground Truth: {len(gt_raw)} 部")
    for mid in gt_raw:
        genres_str = movie_id_to_genres.get(mid, "?")
        print(f"    - {movie_id_to_title.get(mid, mid)} | {genres_str}")

    print(f"\n  YouTubeDNN Top-10 召回:")
    hit_count = 0
    for rank, idx in enumerate(top_indices[:10], 1):
        if idx == 0:
            continue
        raw_mid = all_movie_ids[idx - 1] if idx - 1 < len(all_movie_ids) else None
        if raw_mid is None:
            continue
        title = movie_id_to_title.get(raw_mid, "?")
        genres = movie_id_to_genres.get(raw_mid, "?")
        score = scores[idx]

        in_history = "📖 看过" if raw_mid in hist_raw else ""
        is_hit = "🎯 命中" if raw_mid in gt_raw else ""
        tag = f" [{in_history} {is_hit}]" if (in_history or is_hit) else ""
        if is_hit:
            hit_count += 1
        print(f"    {rank}. {title} | {genres} | score={score:.4f}{tag}")

    print(f"\n  Top-10 中命中 Ground Truth: {hit_count}/{len(gt_raw)}")

# 4. 统计整体召回率
print(f"\n{'=' * 60}")
print("[4/4] 批量统计 HitRate (6040 个测试用户)...")

total_hits_10 = 0
total_hits_20 = 0
for uid in user_ids:
    mask = test["user_id"] == uid
    idx = np.where(mask)[0][0]

    model_inputs = {
        "user_id": test["user_id"][idx].reshape(1, 1),
        "gender": test["gender"][idx].reshape(1, 1),
        "age": test["age"][idx].reshape(1, 1),
        "occupation": test["occupation"][idx].reshape(1, 1),
        "zip_code": test["zip_code"][idx].reshape(1, 1),
        "hist_movie_id": test["hist_movie_id"][idx].reshape(1, 10),
        "hist_genres": test["hist_genres"][idx].reshape(1, 10),
    }

    user_emb = user_model.predict(model_inputs, verbose=0)
    scores = np.dot(user_emb, item_embedding_matrix.T)[0]
    top_indices = np.argsort(scores)[::-1][:20]

    gt = set()
    for enc_mid in user_ground_truth[uid]:
        if enc_mid - 1 < len(all_movie_ids):
            gt.add(all_movie_ids[enc_mid - 1])

    # 编码 ID → 原始 ID 映射
    rec_10 = []
    rec_20 = []
    for i, idx in enumerate(top_indices):
        if idx == 0:
            continue
        raw = all_movie_ids[idx - 1] if idx - 1 < len(all_movie_ids) else None
        if raw is None:
            continue
        if i < 10:
            rec_10.append(raw)
        rec_20.append(raw)

    if gt & set(rec_10):
        total_hits_10 += 1
    if gt & set(rec_20):
        total_hits_20 += 1

n_users = len(user_ids)
print(f"\n  用户总数: {n_users}")
print(f"  HitRate@10: {total_hits_10/n_users*100:.2f}%")
print(f"  HitRate@20: {total_hits_20/n_users*100:.2f}%")
print(f"\n{'=' * 60}")
print("检查完成!")
print("=" * 60)
