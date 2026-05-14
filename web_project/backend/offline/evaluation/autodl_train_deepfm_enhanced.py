"""
AutoDL 独立训练脚本：DeepFM 精排模型（增强版，16 特征）

用法:
    pip install tensorflow==2.15.0 numpy pandas scikit-learn tqdm
    python autodl_train_deepfm_enhanced.py

会自动加载同目录下的 ranking_train_eval_sample.pkl 等文件，
训练完成后保存到 saved_models/ranking_model_enhanced/ 目录。
"""

import os
import sys
import pickle
import time

import numpy as np
import tensorflow as tf

# ── 配置 ──────────────────────────────────────────────────────────────
EMB_DIM = 16              # 嵌入维度
BATCH_SIZE = 512          # 增大 batch 加速训练
EPOCHS = 15
LEARNING_RATE = 0.001

# DeepFM 参数
DNN_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.1

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(DATA_DIR, "saved_models", "ranking_model_enhanced")
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURE_NAMES = [
    # User features (6)
    "user_id", "gender", "age", "occupation", "zip_code", "activity_bucket",
    # Item features (10)
    "movie_id", "genres", "isAdult", "startYear",
    "genre_count", "popularity_bucket", "quality_bucket",
    "runtime_bucket", "movie_age_bucket", "director_bucket",
]

print(f"配置: emb_dim={EMB_DIM}, batch_size={BATCH_SIZE}, epochs={EPOCHS}")
print(f"特征数: {len(FEATURE_NAMES)}")


# ── FM 层实现 ─────────────────────────────────────────────────────────

class FM(tf.keras.layers.Layer):
    """因子分解机 (Factorization Machine) 二阶特征交叉"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: (B, N, D) 每个特征是 embedding 向量
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1))   # (B, D)
        sum_of_square = tf.reduce_sum(tf.square(inputs), axis=1)   # (B, D)
        cross_term = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1, keepdims=True)  # (B, 1)
        return cross_term


class DNNs(tf.keras.layers.Layer):
    """多层 DNN 带 dropout"""
    def __init__(self, units, dropout_rate=0.1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.dense_layers = [tf.keras.layers.Dense(u, activation=activation) for u in units]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = inputs
        for dense in self.dense_layers:
            x = dense(x)
            x = self.dropout(x, training=training)
        return x


# ── 1. 加载数据 ───────────────────────────────────────────────────────
print("\n[1/4] 加载数据...")
data = pickle.load(open(os.path.join(DATA_DIR, "ranking_train_eval_sample.pkl"), "rb"))
feature_dict = pickle.load(open(os.path.join(DATA_DIR, "ranking_feature_dict.pkl"), "rb"))

train_data = data["train"]
test_data = data["test"]

print(f"  训练样本: {len(train_data['is_click']):,}")
print(f"  测试样本: {len(test_data['is_click']):,}")
print(f"  特征词表: {feature_dict}")
print(f"  正样本率 (训练): {train_data['is_click'].mean():.4f}")


# ── 2. 构建 DeepFM 模型 ──────────────────────────────────────────────
print("\n[2/4] 构建 DeepFM 模型 (16 特征)...")

# 输入层
inputs = {}
for name in FEATURE_NAMES:
    inputs[name] = tf.keras.Input(shape=(1,), dtype=tf.int32, name=name)

# Embedding 层（每个特征一个，FM 和 DNN 共享）
emb_layers = {}
for name in FEATURE_NAMES:
    vocab_size = feature_dict[name] + 1  # +1 确保 OOV
    emb_layers[name] = tf.keras.layers.Embedding(
        vocab_size, EMB_DIM, name=f"{name}_embedding"
    )

# 收集所有特征 embedding (B, 1, D)
all_embs = []
for name in FEATURE_NAMES:
    emb = emb_layers[name](inputs[name])  # (B, 1, D)
    all_embs.append(emb)

# FM 二阶交叉
concat_embs = tf.keras.layers.Concatenate(axis=1)(all_embs)  # (B, N, D)
fm_out = FM()(concat_embs)  # (B, 1)

# DNN 隐式交叉
flatten_embs = tf.keras.layers.Flatten()(concat_embs)  # (B, N*D)
dnn_out = DNNs(units=DNN_UNITS, dropout_rate=DROPOUT_RATE, name="dnn_layers")(flatten_embs)
dnn_out = tf.keras.layers.Dense(1, name="dnn_final")(dnn_out)  # (B, 1)

# 线性项 (one-term)
linear_logits = []
for name in FEATURE_NAMES:
    linear_emb = tf.keras.layers.Embedding(
        feature_dict[name] + 1, 1, name=f"{name}_linear"
    )(inputs[name])
    linear_logits.append(tf.keras.layers.Flatten()(linear_emb))
linear_sum = tf.keras.layers.Add()(linear_logits)  # (B, 1)

# 合并 FM + DNN + Linear
combined = tf.keras.layers.Add()([fm_out, dnn_out, linear_sum])  # (B, 1)

# Sigmoid 输出
output = tf.keras.layers.Activation("sigmoid", name="output")(combined)
output = tf.keras.layers.Flatten(name="flatten_output")(output)

model = tf.keras.Model(inputs=inputs, outputs=output, name="deepfm_enhanced")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["binary_accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()
n_params = model.count_params()
print(f"\n  模型参数量: {n_params/1e6:.2f}M")


# ── 3. 准备 tf.data ───────────────────────────────────────────────────
print("\n[3/4] 准备数据...")

def make_tf_dataset(data_dict, batch_size, shuffle=True):
    x = {name: data_dict[name].reshape(-1, 1) for name in FEATURE_NAMES}
    y = data_dict["is_click"].astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_tf_dataset(train_data, BATCH_SIZE, shuffle=True)
test_ds = make_tf_dataset(test_data, BATCH_SIZE * 2, shuffle=False)

n_batches = len(train_data["is_click"]) // BATCH_SIZE
print(f"  每 epoch 步数: {n_batches}")


# ── 4. 训练 ───────────────────────────────────────────────────────────
print(f"\n[4/4] 开始训练 {EPOCHS} epochs...\n")

t_start = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    verbose=1,
)
t_end = time.time()

total_min = (t_end - t_start) / 60
print(f"\n训练完成! 总耗时: {total_min:.1f} 分钟")

# 显示最终结果
final_auc = history.history["val_auc"][-1]
final_loss = history.history["val_loss"][-1]
print(f"\n测试集结果:")
print(f"  Loss: {final_loss:.4f}")
print(f"  AUC:  {final_auc:.4f}")

# 显示所有 epoch 的验证指标
print(f"\n训练过程:")
for epoch in range(EPOCHS):
    print(f"  Epoch {epoch+1:2d}: loss={history.history['loss'][epoch]:.4f}, "
          f"auc={history.history['auc'][epoch]:.4f}, "
          f"val_loss={history.history['val_loss'][epoch]:.4f}, "
          f"val_auc={history.history['val_auc'][epoch]:.4f}")


# ── 5. 保存模型 ───────────────────────────────────────────────────────
print("\n保存模型...")
model.save(SAVE_DIR)

# 保存训练配置和特征信息
pickle.dump({
    "feature_names": FEATURE_NAMES,
    "feature_dict": feature_dict,
    "n_features": len(FEATURE_NAMES),
    "emb_dim": EMB_DIM,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "model_params": {
        "dnn_units": DNN_UNITS,
        "dropout_rate": DROPOUT_RATE,
    },
    "training_history": {
        "loss": [float(x) for x in history.history["loss"]],
        "auc": [float(x) for x in history.history["auc"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_auc": [float(x) for x in history.history["val_auc"]],
    },
    "final_auc": float(final_auc),
}, open(os.path.join(SAVE_DIR, "model_config.pkl"), "wb"))

print(f"  模型保存到: {SAVE_DIR}")
print(f"  config 保存到: {os.path.join(SAVE_DIR, 'model_config.pkl')}")
print("\n完成!")
