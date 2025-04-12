from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

def print_all(title, pd_data, max_columns=None):
    with pd.option_context(
            'display.max_columns', max_columns,
            'display.max_rows', None,
            'display.max_colwidth', None,
            'display.width', 1000,
            'display.expand_frame_repr', False # 禁止自动换行
    ):
        print(f"{title}\n{pd_data}")

movie_data = pd.read_csv("../data/movies.csv")
print_all("movie_data", movie_data.head(15))

movie_data_genres = movie_data.genres.str.get_dummies(sep="|")
print_all("movie_data_genres", movie_data_genres.head(10))
print(f"movie_data_genres size: {movie_data_genres.shape}")

genres_support = apriori(movie_data_genres, min_support=0.025, use_colnames=True).sort_values(by="support", ascending=False)
print_all("genres_support", genres_support.head(20))

genres_lift = association_rules(genres_support, metric="lift", min_threshold=1.25).sort_values(by="lift", ascending=False)
print_all("genres_lift", genres_lift.iloc[:, :10], max_columns=13)

