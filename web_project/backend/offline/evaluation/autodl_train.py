"""
AutoDL 独立训练脚本：YouTubeDNN 召回模型

用法:
    pip install tensorflow==2.15.0 numpy pandas scikit-learn tqdm tabulate
    python autodl_train.py

会自动加载同目录下的 train_eval_sample_final.pkl 等文件，
训练完成后保存到 saved_models/ 目录。
"""

import os
import sys
import pickle
import time

import numpy as np
import tensorflow as tf

# ── 配置 ──────────────────────────────────────────────────────────────
EMB_DIM = 32            # 嵌入维度 (原 16)
MAX_SEQ_LEN = 10
NEG_SAMPLE_SIZE = 50    # 负采样数 (原 20)
BATCH_SIZE = 512        # AutoDL 显存大，可以开大
EPOCHS = 20             # 训练轮数 (原 3)
LEARNING_RATE = 0.001

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(DATA_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"配置: emb_dim={EMB_DIM}, neg_sample={NEG_SAMPLE_SIZE}, "
      f"batch_size={BATCH_SIZE}, epochs={EPOCHS}")


# ── 1. 加载数据 ───────────────────────────────────────────────────────
print("\n[1/4] 加载数据...")
train_eval_samples = pickle.load(
    open(os.path.join(DATA_DIR, "train_eval_sample_final.pkl"), "rb")
)
feature_dict = pickle.load(
    open(os.path.join(DATA_DIR, "feature_dict.pkl"), "rb")
)

train_data = train_eval_samples["train"]
test_data = train_eval_samples["test"]

print(f"  训练样本: {len(train_data['user_id'])}")
print(f"  测试样本: {len(test_data['user_id'])}")
print(f"  物品词表: {feature_dict['movie_id']}")


# ── 2. 构建模型 ───────────────────────────────────────────────────────
print("\n[2/4] 构建 YouTubeDNN 模型...")

# 用户特征输入
user_id = tf.keras.Input(shape=(1,), dtype=tf.int32, name="user_id")
gender = tf.keras.Input(shape=(1,), dtype=tf.int32, name="gender")
age = tf.keras.Input(shape=(1,), dtype=tf.int32, name="age")
occupation = tf.keras.Input(shape=(1,), dtype=tf.int32, name="occupation")
zip_code = tf.keras.Input(shape=(1,), dtype=tf.int32, name="zip_code")
hist_movie_id = tf.keras.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="hist_movie_id")
hist_genres = tf.keras.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="hist_genres")

# ── 共享 Embedding 层（也作为 softmax 权重，即 item embedding 矩阵）──
item_embedding_layer = tf.keras.layers.Embedding(
    input_dim=feature_dict["movie_id"] + 1,   # +1 确保索引不越界
    output_dim=EMB_DIM,
    mask_zero=True,
    name="item_embedding",
)

user_sparse_embeddings = {
    "user_id": tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
        feature_dict["user_id"] + 1, EMB_DIM, name="user_id_embedding")(user_id)),
    "gender": tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
        feature_dict["gender"] + 1, EMB_DIM, name="gender_embedding")(gender)),
    "age": tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
        feature_dict["age"] + 1, EMB_DIM, name="age_embedding")(age)),
    "occupation": tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
        feature_dict["occupation"] + 1, EMB_DIM, name="occupation_embedding")(occupation)),
    "zip_code": tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
        feature_dict["zip_code"] + 1, EMB_DIM, name="zip_code_embedding")(zip_code)),
}

# 历史序列处理（使用共享的 item_embedding_layer 做序列 pooling）
hist_movie_emb = item_embedding_layer(hist_movie_id)            # (B, 10, D)
hist_movie_pooled = tf.reduce_mean(hist_movie_emb, axis=1)      # (B, D)

hist_genre_emb = tf.keras.layers.Embedding(
    feature_dict["genres"] + 1, EMB_DIM, mask_zero=True, name="genres_embedding"
)(hist_genres)
hist_genre_pooled = tf.reduce_mean(hist_genre_emb, axis=1)      # (B, D)

# 拼接用户特征
user_concat = tf.keras.layers.Concatenate()(
    [user_sparse_embeddings["user_id"],
     user_sparse_embeddings["gender"],
     user_sparse_embeddings["age"],
     user_sparse_embeddings["occupation"],
     user_sparse_embeddings["zip_code"],
     hist_movie_pooled,
     hist_genre_pooled]
)  # (B, 7*D)

# 用户 DNN
user_dnn = tf.keras.layers.Dense(128, activation="relu")(user_concat)
user_dnn = tf.keras.layers.Dense(64, activation="relu")(user_dnn)
# 训练用的用户模型（无 L2 归一化 — 让 logits 有足够动态范围）
user_vec = tf.keras.layers.Dense(EMB_DIM, activation=None, name="user_vec")(user_dnn)
user_model_train = tf.keras.Model(
    inputs=[user_id, gender, age, occupation, zip_code, hist_movie_id, hist_genres],
    outputs=user_vec,
    name="user_model_train"
)

# 推理用的用户模型（带 L2 归一化，用于 cosine similarity）
user_vec_norm = tf.keras.layers.Lambda(
    lambda x: tf.math.l2_normalize(x, axis=1), name="l2_normalize"
)(user_vec)
user_model = tf.keras.Model(
    inputs=[user_id, gender, age, occupation, zip_code, hist_movie_id, hist_genres],
    outputs=user_vec_norm,
    name="user_model"
)

# ── 3. 训练准备 ──
print("\n[3/4] 准备训练...")

# 推理用的物品模型（带 L2 归一化）
item_model_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name="movie_id")
item_model_emb = item_embedding_layer(item_model_input)
item_model_emb = tf.keras.layers.Flatten()(item_model_emb)
item_model_output = tf.keras.layers.Lambda(
    lambda x: tf.math.l2_normalize(x, axis=1), name="l2_normalize_item"
)(item_model_emb)
item_model = tf.keras.Model(inputs=item_model_input, outputs=item_model_output, name="item_model")

print("  用户模型 (训练):", user_model_train.output.shape)
print("  用户模型 (推理):", user_model.output.shape)
print("  物品模型 (推理):", item_model.output.shape)

# 优化器 — 提高学习率适应更大模型
LEARNING_RATE = 0.005   # 从 0.001 提升到 0.005
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# 训练数据准备（转换为 tf.data）
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "user_id": train_data["user_id"].reshape(-1, 1),
        "gender": train_data["gender"].reshape(-1, 1),
        "age": train_data["age"].reshape(-1, 1),
        "occupation": train_data["occupation"].reshape(-1, 1),
        "zip_code": train_data["zip_code"].reshape(-1, 1),
        "hist_movie_id": train_data["hist_movie_id"].reshape(-1, MAX_SEQ_LEN),
        "hist_genres": train_data["hist_genres"].reshape(-1, MAX_SEQ_LEN),
    },
    train_data["movie_id"].reshape(-1, 1),  # label
))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(3)

n_batches = len(train_data["user_id"]) // BATCH_SIZE
print(f"  每 epoch 步数: {n_batches}")


# Sampled Softmax 训练步骤
@tf.function
def train_step(user_inputs, pos_movie_ids):
    batch_size = tf.shape(pos_movie_ids)[0]

    with tf.GradientTape() as tape:
        # 用户 embedding（训练模型，无 L2 归一化）
        user_emb = user_model_train(user_inputs, training=True)     # (B, D)

        # 正样本 embedding（无 L2 归一化）
        pos_emb = item_embedding_layer(pos_movie_ids)               # (B, 1, D)
        pos_emb = tf.squeeze(pos_emb, axis=1)                       # (B, D)

        # 正样本 logit = dot(user_emb, pos_emb)  # 无归一化，范围 ~ ±D
        pos_logits = tf.reduce_sum(user_emb * pos_emb, axis=1, keepdims=True)  # (B, 1)

        # 负采样
        num_items = feature_dict["movie_id"]
        sampled_ids = tf.random.uniform(
            [batch_size, NEG_SAMPLE_SIZE],
            minval=1,
            maxval=num_items + 1,
            dtype=tf.int32
        )

        # 负样本 embedding（无 L2 归一化）
        neg_emb = item_embedding_layer(sampled_ids)                 # (B, neg, D)

        # 负样本 logits
        neg_logits = tf.matmul(
            tf.expand_dims(user_emb, axis=1),
            tf.transpose(neg_emb, [0, 2, 1])
        )
        neg_logits = tf.squeeze(neg_logits, axis=1)                 # (B, neg)

        # 拼接正负 logits → softmax
        logits = tf.concat([pos_logits, neg_logits], axis=1)        # (B, 1+neg)
        labels = tf.zeros(batch_size, dtype=tf.int32)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        ))

    # 梯度更新（item_embedding_layer 的权重已含在 user_model_train 中）
    grads = tape.gradient(loss, user_model_train.trainable_weights)
    optimizer.apply_gradients(zip(grads, user_model_train.trainable_weights))

    return loss


# ── 4. 训练 ───────────────────────────────────────────────────────────
print(f"\n[4/4] 开始训练 {EPOCHS} epochs...")
print()

t_start = time.time()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    step_count = 0

    for step, (user_inputs, pos_movie_ids) in enumerate(train_dataset):
        loss = train_step(user_inputs, pos_movie_ids)
        epoch_loss += loss.numpy()
        step_count += 1

    avg_loss = epoch_loss / step_count
    elapsed = time.time() - t_start
    print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  elapsed={elapsed:.0f}s")

t_end = time.time()
total_min = (t_end - t_start) / 60
print(f"\n训练完成! 总耗时: {total_min:.1f} 分钟")


# ── 5. 保存模型 ───────────────────────────────────────────────────────
print("\n保存模型...")
user_model.save(os.path.join(SAVE_DIR, "user_model"))
item_model.save(os.path.join(SAVE_DIR, "item_model"))

# 生成物品 embedding（用于在线推理的 item_embeddings.npy）
print("生成物品 embeddings...")
vocab_dict = pickle.load(open(os.path.join(DATA_DIR, "vocab_dict.pkl"), "rb"))
all_movie_ids = sorted(vocab_dict["movie_id"])
encoded_ids = np.arange(1, len(all_movie_ids) + 1, dtype=np.int32)
item_embs = item_model.predict({"movie_id": encoded_ids}, verbose=0)
item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

np.save(os.path.join(SAVE_DIR, "item_embeddings.npy"), item_embs)
print(f"  模型保存到: {SAVE_DIR}")
print(f"  物品 embeddings: {item_embs.shape}")
print("\n完成!")
