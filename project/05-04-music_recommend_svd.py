import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import sqlite3

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
    return play_data.groupby(by=["user_id", "song_id"])["play_count"].sum() / play_data["user_id"].sum()

def fill_song_info(play_data, song_data, join_column):
    """
    填充歌曲信息
    """
    user_song_data = play_data.merge(song_data, on=join_column, how="left")

    return user_song_data

def build_user_song_score_matrix(data, user_column, item_column):
    """
    构建用户-歌曲评分矩阵
    """