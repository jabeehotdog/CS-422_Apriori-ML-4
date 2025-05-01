import pandas as pd

class Preprocessor:
    def __init__(self, path: str):
        self.df = pd.read_csv(path)

    def preprocess(self, group: str = None) -> pd.DataFrame:
        if group is not None:
            df = self.df[self.df['group'] == group]
        else:
            df = self.df

        interest_cols = [col for col in df.columns if col.startswith('interest')]
        one_hot_df = df[interest_cols].fillna(0).astype(int)
        return one_hot_df

    def get_transactions(self, one_hot_df: pd.DataFrame) -> list:
        transactions = []
        for _, row in one_hot_df.iterrows():
            items = [col for col in one_hot_df.columns if row[col] == 1]
            transactions.append(items)
        return transactions
