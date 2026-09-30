import pandas as pd
import numpy as np

from config import *

def data_load(sessions_path: str, hits_path: str):
    sessions = pd.read_csv(sessions_path, low_memory=False)
    hits = pd.read_csv(hits_path)

    sessions.columns = sessions.columns.str.strip()
    hits.columns = hits.columns.str.strip()

    sessions = sessions.drop_duplicates()
    hits = hits.drop_duplicates()

    def drop_empty_columns(df, threshold=0.95):
        empty_share = df.isna().mean()
        cols_to_drop = empty_share[empty_share >= threshold].index.tolist()
        return df.drop(columns=cols_to_drop)

    sessions = drop_empty_columns(sessions, threshold=0.95)
    hits  = drop_empty_columns(hits, threshold=0.95)

    def inputer(df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].replace("", np.nan).fillna("missing")
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        return df

    sessions = inputer(sessions)
    hits = inputer(hits)

    sessions["visit_date"] = pd.to_datetime(sessions["visit_date"], format='%Y-%m-%d', errors="coerce")
    hits["hit_date"] = pd.to_datetime(hits["hit_date"], format='%Y-%m-%d', errors="coerce")
    sessions["visit_time"] = pd.to_datetime(sessions["visit_time"].astype(str), format='%H:%M:%S',errors="coerce")

    return sessions, hits

def prepare_data(df):
    for col in category_cols:
        if col not in df.columns:
            df[col] = "missing"
        df[col] = df[col].astype(str).fillna("missing")

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "session_id" not in df.columns:
        df = df.dropna(subset=feature_cols)
        return df[feature_cols]
    else:
        df = df.dropna(subset=feature_cols)
        return df[["session_id"] + feature_cols]


def make_target(hits: pd.DataFrame) -> pd.DataFrame:
    target = hits.assign(
        is_target=hits["event_action"].isin(target_action).astype(int)
    ).groupby("session_id", as_index=False)["is_target"].max()
    target = target.rename(columns={"is_target": "target"})
    return target

