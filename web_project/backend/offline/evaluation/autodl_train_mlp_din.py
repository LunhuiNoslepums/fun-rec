"""
AutoDL 独立训练脚本：Deep MLP + DIN 精排模型

架构:
    ┌─ 16 静态特征 → Embedding → Concat/Flatten → DNN([128,64,32]) → MLP logit
    ├─ 16 静态特征 → Linear → Linear logit
    └─ hist_movie_id(seq_len=10) + movie_id → 共享Embedding → DIN Attention → DIN logit
                                                                              │
    sigmoid(MLP + Linear + DIN) ←─────────────────────────────────────────────┘

DIN 注意力机制：
    query = movie_id candidate embedding
    keys  = hist_movie_id (用户行为序列)
    attention score = concat(q, k, q-k, q*k) → FFN([80,40]) → Dense(1) → softmax
    输出 = weighted sum of keys

用法:
    pip install tensorflow==2.15.0 numpy pandas scikit-learn tqdm
    python autodl_train_mlp_din.py
"""

import os
import pickle
import time

import numpy as np
import tensorflow as tf

# ── 配置 ──────────────────────────────────────────────────────────────
EMB_DIM = 16
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.001
DNN_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.1
SEQ_LEN = 10
PATIENCE = 2  # early stopping

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(DATA_DIR, "saved_models", "ranking_model_mlp_din")
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURE_NAMES = [
    # User features (6)
    "user_id", "gender", "age", "occupation", "zip_code", "activity_bucket",
    # Item features (10)
    "movie_id", "genres", "isAdult", "startYear",
    "genre_count", "popularity_bucket", "quality_bucket",
    "runtime_bucket", "movie_age_bucket", "director_bucket",
    # Sequence feature
    "hist_movie_id",
]

print(f"配置: emb_dim={EMB_DIM}, batch_size={BATCH_SIZE}, epochs={EPOCHS}")
print(f"特征数: {len(FEATURE_NAMES)}, 其中序列特征: hist_movie_id")


# ── 自定义层 ──────────────────────────────────────────────────────────

class DinAttentionLayer(tf.keras.layers.Layer):
    """DIN 注意力层: query 与 keys 的加性注意力"""
    def __init__(self, att_units=(80, 40), **kwargs):
        super().__init__(**kwargs)
        self.att_units = att_units
        self.ffn = None

    def build(self, input_shape):
        # input_shape: [query_shape, keys_shape]
        query_dim = input_shape[0][-1]  # D
        # 注意力 FFN: concat(q, k, q-k, q*k) → FFN → score
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(unit, activation="relu")
            for unit in self.att_units
        ] + [tf.keras.layers.Dense(1)])
        super().build(input_shape)

    def call(self, query, keys, mask=None):
        # query: (B, 1, D)
        # keys:  (B, L, D)
        _, L, D = keys.shape
        q = tf.tile(query, [1, L, 1])  # (B, L, D)

        # 经典 DIN 特征交互: concat(q, k, q-k, q*k)
        att_input = tf.concat([
            q,
            keys,
            q - keys,
            q * keys,
        ], axis=-1)  # (B, L, 4*D)

        scores = self.ffn(att_input)  # (B, L, 1)
        scores = tf.squeeze(scores, axis=-1)  # (B, L)

        if mask is not None:
            scores += -1e9 * tf.cast(tf.logical_not(mask), tf.float32)

        att_weights = tf.nn.softmax(scores, axis=-1)  # (B, L)
        att_weights = tf.expand_dims(att_weights, axis=-1)  # (B, L, 1)
        output = tf.reduce_sum(att_weights * keys, axis=1)  # (B, D)
        return output


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
print(f"  hist_movie_id shape: {train_data['hist_movie_id'].shape}")


# ── 2. 构建 Deep MLP + DIN 模型 ──────────────────────────────────────
print("\n[2/4] 构建 Deep MLP + DIN 模型...")

# 输入层
inputs = {}
for name in FEATURE_NAMES:
    if name == "hist_movie_id":
        inputs[name] = tf.keras.Input(shape=(SEQ_LEN,), dtype=tf.int32, name=name)
    else:
        inputs[name] = tf.keras.Input(shape=(1,), dtype=tf.int32, name=name)

# ── 静态特征 Embedding ──
static_names = [n for n in FEATURE_NAMES if n != "hist_movie_id"]
all_static_embs = []
linear_logits = []

for name in static_names:
    vocab_size = feature_dict[name] + 1
    # 共享 embedding 到 DNN
    emb = tf.keras.layers.Embedding(vocab_size, EMB_DIM, name=f"{name}_embedding")(inputs[name])
    all_static_embs.append(emb)
    # Linear 一阶项
    linear_emb = tf.keras.layers.Embedding(vocab_size, 1, name=f"{name}_linear")(inputs[name])
    linear_logits.append(tf.keras.layers.Flatten()(linear_emb))

# ── MLP Tower ──
static_concat = tf.keras.layers.Concatenate(axis=1)(all_static_embs)  # (B, 16, D)
static_flat = tf.keras.layers.Flatten()(static_concat)               # (B, 16*D)
mlp_out = DNNs(units=DNN_UNITS, dropout_rate=DROPOUT_RATE, name="mlp_layers")(static_flat)
mlp_logit = tf.keras.layers.Dense(1, name="mlp_final")(mlp_out)      # (B, 1)

# ── Linear ──
linear_sum = tf.keras.layers.Add()(linear_logits)                    # (B, 1)

# ── DIN Attention ──
# hist_movie_id 与 movie_id 共享 embedding 表
movie_vocab_size = feature_dict["movie_id"] + 1
hist_embedding = tf.keras.layers.Embedding(
    movie_vocab_size, EMB_DIM, name="hist_movie_embedding"
)
# movie_id query: (B, 1) → (B, 1, D)
movie_query = hist_embedding(inputs["movie_id"])  # (B, 1, D)
# hist_movie_id keys: (B, 10) → (B, 10, D)
hist_keys = hist_embedding(inputs["hist_movie_id"])  # (B, 10, D)

# mask: padding 位置 (0) 不参与 attention
hist_mask = tf.cast(inputs["hist_movie_id"] > 0, tf.bool)  # (B, 10)

din_out = DinAttentionLayer(name="din_attention")(movie_query, hist_keys, mask=hist_mask)
din_logit = tf.keras.layers.Dense(64, activation="relu")(din_out)
din_logit = tf.keras.layers.Dense(1, name="din_final")(din_logit)  # (B, 1)

# ── 融合 ──
combined = tf.keras.layers.Add()([mlp_logit, linear_sum, din_logit])  # (B, 1)
output = tf.keras.layers.Activation("sigmoid", name="output")(combined)
output = tf.keras.layers.Flatten(name="flatten_output")(output)

model = tf.keras.Model(inputs=inputs, outputs=output, name="mlp_din")

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
    x = {}
    for name in FEATURE_NAMES:
        if name == "hist_movie_id":
            x[name] = data_dict[name].reshape(-1, SEQ_LEN)
        else:
            x[name] = data_dict[name].reshape(-1, 1)
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


# ── 4. 训练 (带 Early Stopping) ──────────────────────────────────────
print(f"\n[4/4] 开始训练 (最多 {EPOCHS} epochs, early stopping patience={PATIENCE})...\n")

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    """监控 val_auc，连续 patience 个 epoch 无提升则停训"""
    def __init__(self, patience=2):
        super().__init__()
        self.patience = patience
        self.best_auc = 0
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        val_auc = logs.get("val_auc", 0)
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True

early_stop = EarlyStoppingCallback(patience=PATIENCE)

t_start = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=[early_stop],
    verbose=1,
)
t_end = time.time()

# 恢复最佳权重
if early_stop.best_weights is not None:
    model.set_weights(early_stop.best_weights)
    print(f"\n恢复 epoch {early_stop.best_epoch} 的最佳权重 (val_auc={early_stop.best_auc:.4f})")

total_min = (t_end - t_start) / 60
print(f"\n训练完成! 总耗时: {total_min:.1f} 分钟")
if early_stop.stopped_epoch > 0:
    print(f"  Early stopping 于 epoch {early_stop.stopped_epoch}")

# 最终评估
final_auc = early_stop.best_auc
final_loss = min(history.history["val_loss"])
print(f"\n测试集结果:")
print(f"  Best val_auc:  {final_auc:.4f} (epoch {early_stop.best_epoch})")
print(f"  Best val_loss: {final_loss:.4f}")

# 显示所有 epoch 的验证指标
print(f"\n训练过程:")
for epoch in range(len(history.history["val_auc"])):
    print(f"  Epoch {epoch+1:2d}: loss={history.history['loss'][epoch]:.4f}, "
          f"auc={history.history['auc'][epoch]:.4f}, "
          f"val_loss={history.history['val_loss'][epoch]:.4f}, "
          f"val_auc={history.history['val_auc'][epoch]:.4f}")


# ── 5. 保存模型 ───────────────────────────────────────────────────────
print("\n保存模型...")
model.save(SAVE_DIR)

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
        "seq_len": SEQ_LEN,
        "patience": PATIENCE,
    },
    "best_epoch": early_stop.best_epoch,
    "best_val_auc": float(final_auc),
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
