import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sqlite3
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import svds
from utils.model.score_by_svd import ScoringBySVD

def load_play_data(path=None):
    """
    加载播放数据
    """
    if path is None:
        path = "../data/train_triplets.txt"
    
    data = pd.read_csv(path, sep="\t", names=["user_id", "song_id", "play_count"
    ])

    return data

def read_song_info(table_name, song_columns, song_db=None):
    """
    读取歌曲数据库的指定字段，并转换成dataframe对象
    """
    if song_db is None:
        song_db = "../data/track_metadata.db"
    
    conn = sqlite3.connect(song_db)
    cursor = conn.cursor()
    sql = f"select {', '.join(song_columns)} from {table_name}"
    song_df = pd.read_sql_query(sql, conn)

    conn.close()

    return song_df

def calc_song_score(play_data):
    """
    计算当前用户歌单中的歌曲得分
    """
    # 1. 转换字符串列为分类类型 (减少内存+加速分组)
    play_data["user_id"] = play_data["user_id"].astype("category")
    play_data["song_id"] = play_data["song_id"].astype("category")

    # 2. 按用户+歌曲分组求和
    grouped = play_data.groupby(["user_id", "song_id"], as_index=False, observed=True)["play_count"].sum()
    
    # 3. 计算用户级总播放次数
    grouped["user_total"] = grouped.groupby("user_id", observed=True)["play_count"].transform("sum")
    
    # 4. 计算占比
    grouped["score"] = grouped["play_count"] / grouped["user_total"]
    
    return grouped[["user_id", "song_id", "score"]]

def fill_song_info(play_data, song_data, join_column):
    """
    填充歌曲信息
    """
    # 预处理歌曲数据
    song_lookup = (song_data
                   .drop_duplicates(join_column)  # 确保键唯一
                   .set_index(join_column)  # 建立索引
                   .astype({"title": "category", "release": "category", 
                           "artist_name": "category"}))  # 类型优化
    
    # 分列映射避免内存溢出
    play_data["title"] = play_data[join_column].map(song_lookup["title"])
    play_data["release"] = play_data[join_column].map(song_lookup["release"])
    play_data["artist_name"] = play_data[join_column].map(song_lookup["artist_name"])
    play_data["artist_familiarity"] = play_data[join_column].map(song_lookup["artist_familiarity"])
    play_data["artist_hotttnesss"] = play_data[join_column].map(song_lookup["artist_hotttnesss"])
    
    return play_data

def move_column(data, column_name):
    columns = data.columns.tolist()
    columns.append(columns.pop(columns.index(column_name)))

    return data[columns]

def create_label_encoder(data, column_name_list):
    """
    创建将user_id和song_id编码成数字的labelEncoder
    """
    encoder_map = {}
    for column_name in column_name_list:
        clean_data = data[column_name].dropna().astype(str)
        unique_values = clean_data.unique()

        encoder = LabelEncoder()
        encoder.fit(unique_values)
        encoder_map[column_name] = encoder
    
    return encoder_map

def build_user_song_score_matrix(data, index, column, value):
    """
    构建用户-歌曲评分矩阵
    """
    rows = data[index].values
    cols = data[column].values
    scores = data[value].values

    # 获取矩阵维度
    n_users = data[index].max() + 1
    n_items = data[column].max() + 1
    

    return csr_matrix((scores, (rows, cols)), shape=(n_users, n_items))

def svd_divide(data, n_components):
    """
    将矩阵进行svd分解
    """
    U, s, Vt = svds(data, n_components)
    S = np.diag(s)

    return U, S, Vt

def get_recomendation(user_id, U, S, Vt, user_encoder, song_encoder, used_items, recommd_num=10):
    """
    获取用户推荐列表
    """
    user_encoded_id = user_encoder.transform([user_id])[0]
    item_scores = U[user_encoded_id] @ S @ Vt

    used_items_encoded_id = song_encoder.transform(used_items)
    item_scores[used_items_encoded_id] = 0

    recommend_items = np.argsort(item_scores)[::-1][:recommd_num]

    orig_item_id = song_encoder.inverse_transform(recommend_items)

    return orig_item_id

def load_clean_data(path=None):
    """
    加载数据并进行清理
    """
    action_score_data = calc_song_score(load_play_data(path))
    
    label_encoder = create_label_encoder(action_score_data, ["user_id", "song_id"])
    action_score_data["user_id"] = label_encoder["user_id"].transform(action_score_data["user_id"])
    action_score_data["song_id"] = label_encoder["song_id"].transform(action_score_data["song_id"])

    score_matrix = build_user_song_score_matrix(action_score_data, "user_id", "song_id", "score")

    return score_matrix, label_encoder



if "__main__" == __name__:
    score_matrix, label_encoder = load_clean_data()
    svd_transformer = ScoringBySVD(n_components=50)
    svd_transformer.fit(score_matrix)