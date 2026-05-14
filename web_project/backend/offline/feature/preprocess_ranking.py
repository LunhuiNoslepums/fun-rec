"""
精排模型数据预处理 (DeepFM 等)

本模块为精排/CTR 预估生成逐点样本：
- 正样本：用户-物品交互，click=True 或 conversion=True
- 困难负样本：曝光但未点击/转化 (exposure=True, click=False, conversion=False)
- 随机负样本：用户从未交互过的物品

输出格式：
{
    "train": {
        "user_id": [...], "gender": [...], ...,  # 用户特征
        "movie_id": [...], "genres": [...], ..., # 物品特征
        "is_click": [0, 1, 1, 0, ...]             # 二分类标签
    },
    "test": { ... }
}
"""

import sys
import pickle
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 分桶辅助常量
current_year = 2026

from offline.config import config


def load_raw_data():
    """加载原始 MovieLens 数据文件"""
    print("加载原始数据...")
    try:
        df_movies = pd.read_pickle(config.DATASET_DIR / "movies.pkl")
        df_ratings = pd.read_pickle(config.DATASET_DIR / "ratings.pkl")
        df_users = pd.read_pickle(config.DATASET_DIR / "users.pkl")
        # 新增：加载电影元数据（导演信息）
        movie_metadata_path = config.DATASET_DIR / "movie_metadata.pkl"
        if movie_metadata_path.exists():
            with open(movie_metadata_path, "rb") as f:
                movie_metadata = pickle.load(f)
            df_title_crew = movie_metadata.get("title_crew")
        else:
            df_title_crew = None
        return df_movies, df_ratings, df_users, df_title_crew
    except FileNotFoundError:
        print(f"数据文件不存在: {config.DATASET_DIR}")
        sys.exit(1)


def process_features_for_ranking(df_movies, df_ratings, df_users, df_title_crew=None):
    """
    为精排模型处理特征

    Returns:
        df_merged: 包含所有特征和标签的 DataFrame
        user_vocab: 用户特征词表
        movie_vocab: 电影特征词表
    """
    print("处理特征...")

    # 选择列
    user_columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    movie_columns = ["movie_id", "genres", "isAdult", "startYear",
                     "averageRating", "numVotes", "runtimeMinutes", "imdb_id"]
    ratings_columns = ["user_id", "movie_id", "rating", "timestamp"]

    df_users = df_users[user_columns].copy()
    df_movies = df_movies[movie_columns].copy()

    # 新增: 统计类型数量（在分割前）
    df_movies['genre_count'] = df_movies['genres'].str.split('|').str.len()

    # 处理类型 - 为简化精排取第一个类型
    # (DeepFM 期望标量特征，而非序列)
    df_movies['genres'] = df_movies['genres'].str.split("|").str[0]
    df_movies['isAdult'] = df_movies['isAdult'].fillna(False)
    df_movies['startYear'] = df_movies['startYear'].fillna(0)

    # 新增: 处理数值特征并分桶
    # popularity_bucket (基于 numVotes)
    df_movies['numVotes'] = pd.to_numeric(df_movies['numVotes'], errors='coerce').fillna(0)
    pop_bins = [-1, 0, 100, 1000, 10000, 100000, float('inf')]
    pop_labels = ['unknown', 'very_low', 'low', 'medium', 'high', 'very_high']
    df_movies['popularity_bucket'] = pd.cut(
        df_movies['numVotes'], bins=pop_bins, labels=pop_labels, right=True
    )

    # quality_bucket (基于 averageRating)
    df_movies['averageRating'] = df_movies['averageRating'].fillna(0)
    qual_bins = [-1, 0, 4, 5, 6, 7, 10]
    qual_labels = ['unknown', 'low', 'medium_low', 'medium', 'high', 'very_high']
    df_movies['quality_bucket'] = pd.cut(
        df_movies['averageRating'], bins=qual_bins, labels=qual_labels, right=True
    )

    # runtime_bucket
    df_movies['runtimeMinutes'] = pd.to_numeric(df_movies['runtimeMinutes'], errors='coerce').fillna(0)
    runtime_bins = [-1, 1, 60, 90, 120, 180, float('inf')]
    runtime_labels = ['unknown', 'short', 'medium', 'long', 'very_long', 'epic']
    df_movies['runtime_bucket'] = pd.cut(
        df_movies['runtimeMinutes'], bins=runtime_bins, labels=runtime_labels, right=True
    )

    # movie_age_bucket (startYear → 2026 年差)
    df_movies['startYear_num'] = pd.to_numeric(df_movies['startYear'], errors='coerce').fillna(current_year)
    df_movies['movie_age'] = current_year - df_movies['startYear_num']
    age_bins = [-1, 0, 5, 10, 20, 30, float('inf')]
    age_labels = ['unknown', 'new', 'recent', 'moderate', 'old', 'classic']
    df_movies['movie_age_bucket'] = pd.cut(
        df_movies['movie_age'], bins=age_bins, labels=age_labels, right=True
    )

    # director_bucket (基于导演在数据集中作品数)
    if df_title_crew is not None:
        df_title_crew['directors'] = df_title_crew['directors'].fillna('')
        all_directors = df_title_crew['directors'].str.split(',').explode()
        all_directors = all_directors[all_directors != '']
        director_freq = all_directors.value_counts().to_dict()
        director_map = df_title_crew[['tconst', 'directors']].copy()
        director_map['first_director'] = director_map['directors'].str.split(',').str[0]
        director_map.loc[director_map['first_director'] == '', 'first_director'] = None
        director_map['director_movie_count'] = director_map['first_director'].map(director_freq).fillna(0)
        # 合并到电影表
        df_movies = df_movies.merge(
            director_map[['tconst', 'director_movie_count']],
            left_on='imdb_id', right_on='tconst', how='left'
        )
    else:
        df_movies['director_movie_count'] = 0
    df_movies['director_movie_count'] = df_movies['director_movie_count'].fillna(0)
    dir_bins = [-1, 0, 1, 2, 5, 20, float('inf')]
    dir_labels = ['unknown', 'new', 'occasional', 'regular', 'prolific', 'top']
    df_movies['director_bucket'] = pd.cut(
        df_movies['director_movie_count'], bins=dir_bins, labels=dir_labels, right=True
    )

    df_ratings = df_ratings[ratings_columns].copy()
    
    # 编码用户特征
    print("编码用户特征...")
    user_vocab = {}
    # 新增: activity_bucket 从 ratings 表计算
    user_activity = df_ratings.groupby('user_id').size().reset_index(name='rating_count')
    df_users = df_users.merge(user_activity, on='user_id', how='left')
    df_users['rating_count'] = df_users['rating_count'].fillna(0)
    act_bins = [-1, 50, 200, 500, float('inf')]
    act_labels = ['low', 'medium', 'high', 'very_high']
    df_users['activity_bucket'] = pd.cut(
        df_users['rating_count'], bins=act_bins, labels=act_labels, right=True
    )

    user_sparse_feature_columns = ["user_id", "gender", "age", "occupation", "zip_code",
                                    "activity_bucket"]
    
    for feat_name in user_sparse_feature_columns:
        label_encoder = LabelEncoder()
        df_users[feat_name + "_encoded"] = label_encoder.fit_transform(df_users[feat_name]) + 1  # 0 用于填充/未知
        user_vocab[feat_name] = label_encoder.classes_
    
    # 编码电影特征
    print("编码电影特征...")
    movie_vocab = {}
    movie_sparse_feature_columns = ["movie_id", "genres", "isAdult", "startYear",
                                     "genre_count", "popularity_bucket", "quality_bucket",
                                     "runtime_bucket", "movie_age_bucket", "director_bucket"]

    for feat_name in movie_sparse_feature_columns:
        label_encoder = LabelEncoder()
        # 处理潜在的 NaN 值
        if feat_name in df_movies.columns:
            df_movies[feat_name] = df_movies[feat_name].fillna("unknown" if df_movies[feat_name].dtype == object else 0)
            df_movies[feat_name + "_encoded"] = label_encoder.fit_transform(df_movies[feat_name].astype(str)) + 1
        else:
            print(f"  ⚠ 特征 {feat_name} 不在数据中，使用默认值 1")
            df_movies[feat_name + "_encoded"] = 1
            label_encoder.classes_ = np.array(["unknown"])
        movie_vocab[feat_name] = label_encoder.classes_
    
    # 计算用户平均评分，用于生成标签
    print("计算用户平均评分...")
    user_avg_ratings = df_ratings.groupby("user_id")["rating"].mean().reset_index()
    user_avg_ratings.columns = ["user_id", "user_avg_rating"]
    df_ratings = df_ratings.merge(user_avg_ratings, on="user_id", how="left")
    
    # 基于已有逻辑生成标签：
    # - conversion（转化）: rating >= user_avg_rating（强正向信号）
    # - click（点击）: rating >= user_avg_rating - 1（弱正向信号）
    # - exposure（曝光）: 所有交互
    df_ratings['conversion'] = (df_ratings['rating'] >= df_ratings['user_avg_rating']).astype(int)
    df_ratings['click'] = (df_ratings['rating'] >= df_ratings['user_avg_rating'] - 1).astype(int)
    df_ratings['exposure'] = 1  # 所有评分都是曝光
    
    # 对于精排，使用 click 作为主要标签
    # 正样本: click=1
    # 困难负样本（来自曝光）: click=0
    df_ratings['is_click'] = df_ratings['click']
    
    # 合并所有特征
    print("合并特征...")
    user_merge_cols = ["user_id", "user_id_encoded", "gender_encoded", "age_encoded",
                       "occupation_encoded", "zip_code_encoded", "activity_bucket_encoded"]
    df_merged = df_ratings.merge(
        df_users[user_merge_cols],
        on="user_id",
        how="left"
    )

    movie_merge_cols = ["movie_id", "movie_id_encoded", "genres_encoded",
                        "isAdult_encoded", "startYear_encoded",
                        "genre_count_encoded", "popularity_bucket_encoded",
                        "quality_bucket_encoded", "runtime_bucket_encoded",
                        "movie_age_bucket_encoded", "director_bucket_encoded"]
    df_merged = df_merged.merge(
        df_movies[movie_merge_cols],
        on="movie_id",
        how="left"
    )

    # 重命名编码列为最终名称
    df_merged = df_merged.rename(columns={
        "user_id_encoded": "user_id_enc",
        "gender_encoded": "gender",
        "age_encoded": "age",
        "occupation_encoded": "occupation",
        "zip_code_encoded": "zip_code",
        "activity_bucket_encoded": "activity_bucket",
        "movie_id_encoded": "movie_id_enc",
        "genres_encoded": "genres",
        "isAdult_encoded": "isAdult",
        "startYear_encoded": "startYear",
        "genre_count_encoded": "genre_count",
        "popularity_bucket_encoded": "popularity_bucket",
        "quality_bucket_encoded": "quality_bucket",
        "runtime_bucket_encoded": "runtime_bucket",
        "movie_age_bucket_encoded": "movie_age_bucket",
        "director_bucket_encoded": "director_bucket",
    })

    # 保留原始 ID 用于负采样，编码后的 ID 用于模型
    df_merged["user_id_original"] = df_merged["user_id"]
    df_merged["movie_id_original"] = df_merged["movie_id"]
    df_merged["user_id"] = df_merged["user_id_enc"]
    df_merged["movie_id"] = df_merged["movie_id_enc"]

    # ── 构建用户行为序列 (hist_movie_id) ──
    print("构建用户行为序列 (hist_movie_id, max_len=10)...")
    df_merged = df_merged.sort_values(["user_id_original", "timestamp"])

    max_seq_len = 10
    padding_value = 0
    user_histories = {}
    hist_movie_id_list = []

    for idx, row in df_merged.iterrows():
        uid = row["user_id_original"]
        mid = row["movie_id"]  # 已经赋值为 encoded movie_id

        # 当前记录的历史 = 之前看过的所有电影（排除当前电影，防 label leakage）
        history = user_histories.get(uid, [])

        # 左 padding 到 max_seq_len
        if len(history) >= max_seq_len:
            padded = history[-max_seq_len:]
        else:
            padded = [padding_value] * (max_seq_len - len(history)) + history
        hist_movie_id_list.append(padded)

        # 将当前电影加入用户历史（供下条记录使用）
        user_histories.setdefault(uid, []).append(mid)

    df_merged["hist_movie_id"] = hist_movie_id_list

    return df_merged, user_vocab, movie_vocab


def generate_negative_samples(
    df_merged, 
    movie_vocab,
    neg_ratio_from_exposure=1,  # 每个正样本的困难负样本比例
    neg_ratio_random=2,         # 每个正样本的随机负样本比例
):
    """
    为精排生成负样本
    
    困难负样本按用户采样：对于每个用户的正样本，
    从该用户曝光但未点击的物品中采样。
    这使它们成为"困难"样本，因为用户实际看到了物品但选择不参与。
    
    随机负样本是用户从未交互过的物品。
    
    Args:
        df_merged: 包含正样本和曝光样本的 DataFrame
        movie_vocab: 电影特征词表
        neg_ratio_from_exposure: 每个正样本的困难负样本数量
        neg_ratio_random: 每个正样本的随机负样本数量
    
    Returns:
        添加了负样本的 DataFrame
    """
    print("生成负样本...")
    
    # 用于随机采样的所有编码后电影 ID
    all_movie_ids = set(range(1, len(movie_vocab["movie_id"]) + 1))
    
    # 获取用于负采样的电影特征
    movie_features = df_merged[["movie_id", "genres", "isAdult", "startYear",
                                 "genre_count", "popularity_bucket", "quality_bucket",
                                 "runtime_bucket", "movie_age_bucket", "director_bucket",
                                 "movie_id_original"]].drop_duplicates()
    movie_features_dict = movie_features.set_index("movie_id").to_dict("index")
    
    # 分离正样本和困难负样本（曝光但未点击）
    positive_samples = df_merged[df_merged["is_click"] == 1].copy()
    hard_negative_pool = df_merged[df_merged["is_click"] == 0].copy()
    
    print(f"  正样本: {len(positive_samples)}")
    print(f"  困难负样本池 (曝光但未点击): {len(hard_negative_pool)}")
    
    # 构建每个用户的困难负样本池
    # Key: user_id_original, Value: 该用户的困难负样本 DataFrame
    user_hard_negatives = {}
    for user_id, group in hard_negative_pool.groupby("user_id_original"):
        user_hard_negatives[user_id] = group
    
    # 构建每个用户的交互历史（用于排除随机负样本）
    user_interactions = df_merged.groupby("user_id_original")["movie_id"].apply(set).to_dict()
    
    # 统计同时有正样本和困难负样本的用户数
    users_with_hard_neg = set(user_hard_negatives.keys())
    positive_users = set(positive_samples["user_id_original"].unique())
    users_with_both = users_with_hard_neg & positive_users
    print(f"  同时有正样本和困难负样本的用户数: {len(users_with_both)}")
    
    negative_samples = []
    
    # 1. 按用户采样困难负样本（曝光但未点击）
    if neg_ratio_from_exposure > 0:
        print(f"  为每个正样本采样 {neg_ratio_from_exposure} 个困难负样本...")
        hard_neg_list = []
        hard_neg_count = 0
        
        # 按用户分组正样本以提高处理效率
        for user_id, user_positives in tqdm(positive_samples.groupby("user_id_original"), 
                                             desc="困难负样本 (每个用户)"):
            # 获取该用户的困难负样本池
            user_hard_neg_pool = user_hard_negatives.get(user_id)
            
            if user_hard_neg_pool is None or len(user_hard_neg_pool) == 0:
                continue
            
            # 计算该用户需要采样的困难负样本数
            n_positives = len(user_positives)
            n_hard_neg_needed = n_positives * neg_ratio_from_exposure
            
            # 从该用户的困难负样本池中采样
            n_to_sample = min(len(user_hard_neg_pool), n_hard_neg_needed)
            if n_to_sample > 0:
                sampled = user_hard_neg_pool.sample(
                    n=n_to_sample, 
                    replace=len(user_hard_neg_pool) < n_to_sample
                )
                hard_neg_list.append(sampled)
                hard_neg_count += len(sampled)
        
        if hard_neg_list:
            hard_neg_df = pd.concat(hard_neg_list, ignore_index=True)
            hard_neg_df["is_click"] = 0
            negative_samples.append(hard_neg_df)
            print(f"    添加了 {hard_neg_count} 个困难负样本")
    
    # 2. 生成随机负样本（用户从未交互过的物品）
    if neg_ratio_random > 0:
        print(f"  为每个正样本采样 {neg_ratio_random} 个随机负样本...")
        random_neg_list = []
        
        for _, row in tqdm(positive_samples.iterrows(), total=len(positive_samples), 
                          desc="随机负样本"):
            user_id_orig = row["user_id_original"]
            user_interacted = user_interactions.get(user_id_orig, set())
            
            # 获取用户未交互过的电影 ID
            available_movies = list(all_movie_ids - user_interacted)
            
            if len(available_movies) < neg_ratio_random:
                continue
                
            # 采样随机负样本
            neg_movie_ids = random.sample(available_movies, neg_ratio_random)
            
            for neg_movie_id in neg_movie_ids:
                if neg_movie_id in movie_features_dict:
                    movie_feats = movie_features_dict[neg_movie_id]
                    random_neg_list.append({
                        "user_id": row["user_id"],
                        "user_id_original": user_id_orig,
                        "gender": row["gender"],
                        "age": row["age"],
                        "occupation": row["occupation"],
                        "zip_code": row["zip_code"],
                        "activity_bucket": row["activity_bucket"],
                        "hist_movie_id": row["hist_movie_id"],
                        "movie_id": neg_movie_id,
                        "movie_id_original": movie_feats.get("movie_id_original", neg_movie_id),
                        "genres": movie_feats.get("genres", 0),
                        "isAdult": movie_feats.get("isAdult", 0),
                        "startYear": movie_feats.get("startYear", 0),
                        "genre_count": movie_feats.get("genre_count", 1),
                        "popularity_bucket": movie_feats.get("popularity_bucket", 0),
                        "quality_bucket": movie_feats.get("quality_bucket", 0),
                        "runtime_bucket": movie_feats.get("runtime_bucket", 0),
                        "movie_age_bucket": movie_feats.get("movie_age_bucket", 0),
                        "director_bucket": movie_feats.get("director_bucket", 0),
                        "is_click": 0,
                        "rating": 0,
                        "timestamp": row["timestamp"],
                    })
        
        if random_neg_list:
            random_neg_df = pd.DataFrame(random_neg_list)
            negative_samples.append(random_neg_df)
            print(f"    添加了 {len(random_neg_df)} 随机负样本")
    
    # 合并正样本和负样本
    output_cols = ["user_id", "gender", "age", "occupation", "zip_code",
                   "activity_bucket",
                   "movie_id", "genres", "isAdult", "startYear",
                   "genre_count", "popularity_bucket", "quality_bucket",
                   "runtime_bucket", "movie_age_bucket", "director_bucket",
                   "hist_movie_id",
                   "is_click", "timestamp", "user_id_original"]
    all_samples = [positive_samples[output_cols]]
    
    for neg_df in negative_samples:
        # 确保所有必需列存在
        for col in output_cols:
            if col not in neg_df.columns:
                neg_df[col] = 0
        all_samples.append(neg_df[output_cols])
    
    df_final = pd.concat(all_samples, ignore_index=True)
    
    # 确保所有列为整数类型 (除 hist_movie_id 是序列外)
    feature_cols = ["user_id", "gender", "age", "occupation", "zip_code",
                    "activity_bucket",
                    "movie_id", "genres", "isAdult", "startYear",
                    "genre_count", "popularity_bucket", "quality_bucket",
                    "runtime_bucket", "movie_age_bucket", "director_bucket",
                    "is_click"]
    for col in feature_cols:
        df_final[col] = df_final[col].fillna(0).astype(int)
    
    print(f"  最终数据集: {len(df_final)} 样本")
    print(f"    正样本: {(df_final['is_click'] == 1).sum()}")
    print(f"    负样本 (总数): {(df_final['is_click'] == 0).sum()}")
    
    return df_final


def split_train_test(df_final, test_ratio=0.2, by_time=True):
    """
    将数据划分为训练集和测试集
    
    Args:
        df_final: 包含所有样本的 DataFrame
        test_ratio: 测试集比例
        by_time: 如果为 True，使用时间划分；否则随机划分
    
    Returns:
        train_df, test_df
    """
    print("划分训练集和测试集...")
    
    if by_time and "timestamp" in df_final.columns:
        # 时间划分：使用较新的数据作为测试集
        df_final = df_final.sort_values("timestamp")
        split_idx = int(len(df_final) * (1 - test_ratio))
        train_df = df_final.iloc[:split_idx]
        test_df = df_final.iloc[split_idx:]
    else:
        # 随机划分
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_final, test_size=test_ratio, random_state=42)
    
    print(f"  训练集: {len(train_df)} 样本 (正样本: {(train_df['is_click']==1).sum()})")
    print(f"  测试集: {len(test_df)} 样本 (正样本: {(test_df['is_click']==1).sum()})")
    
    return train_df, test_df


def convert_to_dict(df, feature_columns, label_column="is_click"):
    """将 DataFrame 转换为用于模型训练的字典格式"""
    result = {}
    for col in feature_columns:
        if col == "hist_movie_id":
            # 序列列：已经是列表形式，直接转为 2D numpy array
            result[col] = np.stack(df[col].values).astype(np.int32)
        else:
            result[col] = df[col].values.astype(np.int32)
    result[label_column] = df[label_column].values.astype(np.int32)

    # 保留 user_id_original 用于 gAUC 评估
    if "user_id_original" in df.columns:
        result["user_id_original"] = df["user_id_original"].values

    return result


def run_ranking_preprocessing(
    neg_ratio_from_exposure=1,
    neg_ratio_random=2,
    test_ratio=0.2
):
    """
    精排模型预处理主流程
    
    Args:
        neg_ratio_from_exposure: 曝光困难负样本比例
        neg_ratio_random: 随机负样本比例
        test_ratio: 测试集比例
    """
    print("=" * 60)
    print("精排模型数据预处理")
    print("=" * 60)
    
    # 1. 加载原始数据
    df_movies, df_ratings, df_users, df_title_crew = load_raw_data()

    # 2. 处理特征
    df_merged, user_vocab, movie_vocab = process_features_for_ranking(
        df_movies, df_ratings, df_users, df_title_crew
    )
    
    # 3. 生成负样本
    df_final = generate_negative_samples(
        df_merged,
        movie_vocab,
        neg_ratio_from_exposure=neg_ratio_from_exposure,
        neg_ratio_random=neg_ratio_random,
    )
    
    # 4. 划分训练集和测试集
    train_df, test_df = split_train_test(df_final, test_ratio=test_ratio, by_time=True)
    
    # 5. 转换为字典格式
    feature_columns = ["user_id", "gender", "age", "occupation", "zip_code",
                        "activity_bucket",
                        "movie_id", "genres", "isAdult", "startYear",
                        "genre_count", "popularity_bucket", "quality_bucket",
                        "runtime_bucket", "movie_age_bucket", "director_bucket",
                        "hist_movie_id"]
    
    train_data = convert_to_dict(train_df, feature_columns, "is_click")
    test_data = convert_to_dict(test_df, feature_columns, "is_click")
    
    samples = {"train": train_data, "test": test_data}
    
    # 6. 构建特征词典 (嵌入词汇表大小)
    vocab_dict = {**user_vocab, **movie_vocab}
    feature_dict = {k: len(v) + 1 for k, v in vocab_dict.items()}  # +1 for padding/unknown
    
    # 7. 保存输出
    print("保存处理后的数据...")
    ranking_data_path = config.TEMP_DIR / "ranking_train_eval_sample.pkl"
    ranking_feature_dict_path = config.TEMP_DIR / "ranking_feature_dict.pkl"
    ranking_vocab_dict_path = config.TEMP_DIR / "ranking_vocab_dict.pkl"
    
    pickle.dump(samples, open(ranking_data_path, "wb"))
    pickle.dump(feature_dict, open(ranking_feature_dict_path, "wb"))
    pickle.dump(vocab_dict, open(ranking_vocab_dict_path, "wb"))
    
    print(f"\n保存到:")
    print(f"  - {ranking_data_path}")
    print(f"  - {ranking_feature_dict_path}")
    print(f"  - {ranking_vocab_dict_path}")
    
    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)
    
    # Print summary
    print("\n数据摘要:")
    print(f"  特征: {feature_columns}")
    print(f"  特征词汇表大小: {feature_dict}")
    print(f"  训练集样本: {len(train_data['user_id'])}")
    print(f"  测试集样本: {len(test_data['user_id'])}")
    print(f"  训练集正样本比例: {train_data['is_click'].mean():.2%}")
    print(f"  测试集正样本比例: {test_data['is_click'].mean():.2%}")
    
    return samples, feature_dict


if __name__ == "__main__":
    run_ranking_preprocessing()
