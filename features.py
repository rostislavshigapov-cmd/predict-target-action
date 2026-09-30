import pandas as pd

from config import feature_cols
from data_loading import prepare_data


def build_features(session: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    df = session.copy()
    hit = hits.copy()

    df['year'] = df['visit_date'].dt.year
    df['month'] = df['visit_date'].dt.month
    df['day'] = df['visit_date'].dt.day
    df['is_weekend'] = df['visit_date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['hour'] = df['visit_time'].dt.hour
    df["is_returning"] = (df["visit_number"] > 1).astype(int)

    hit['car_brand'] = hit.hit_page_path.apply(lambda x: x.split('/')[3] if '/cars/all/' in x else 'missing')
    hit['car_model'] = hit.hit_page_path.apply(lambda x: x.split('/')[4] if '/cars/all/' in x else 'missing')
    hit['car_brand'] = hit['car_brand'].fillna('missing')
    hit['car_model'] = hit['car_model'].fillna('missing')

    hit_features = (
        hit.groupby("session_id").agg(
            hit_cnt = ("hit_number", "count"),
            unique_pages = ("hit_page_path", "nunique")
        )
    )

    df = df.merge(hit_features, on="session_id", how="left")
    df = df.merge(hit[['session_id','car_brand', 'car_model']], on="session_id", how="left")

    final_df = prepare_data(df)

    return final_df[['session_id'] + feature_cols]

