import pandas as pd
import json
import argparse


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw interest-group CSV into a DataFrame.
    """
    df = pd.read_csv(path)
    return df


def preprocess_for_apriori(df: pd.DataFrame, group: str = None) -> pd.DataFrame:
    """
    Filter by group (if provided) and produce a one-hot encoded DataFrame for Apriori.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with columns: 'group', 'grand_tot_interests', 'interest1'..'interest217'.
    group : str, optional
        If specified, only include rows from this 'group' category.

    Returns
    -------
    one_hot_df : pd.DataFrame
        DataFrame of shape (n_transactions, n_interests) with 0/1 values.
    """
    if group is not None:
        df = df[df['group'] == group]

    # Identify all interest columns
    interest_cols = [col for col in df.columns if col.startswith('interest')]

    # Fill NaN as 0, convert to int for one-hot representation
    one_hot_df = df[interest_cols].fillna(0).astype(int)
    one_hot_df.columns = [col for col in interest_cols]
    return one_hot_df


def get_transactions(one_hot_df: pd.DataFrame) -> list:
    """
    Convert one-hot encoded DataFrame into a list of transactions (list of item strings).

    Parameters
    ----------
    one_hot_df : pd.DataFrame
        One-hot encoded DataFrame of 0/1 values.

    Returns
    -------
    transactions : list of list of str
        Each inner list is the set of interest-columns where value == 1.
    """
    transactions = []
    for idx, row in one_hot_df.iterrows():
        items = [col for col in one_hot_df.columns if row[col] == 1]
        transactions.append(items)
    return transactions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for Apriori Algorithm (one-hot encode and/or extract transactions)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-csv", type=str, required=True,
        help="Path to raw interest-group CSV file"
    )
    parser.add_argument(
        "--filter-group", type=str, default=None,
        help="Only include rows from this 'group' category (e.g., C, P, R, I)"
    )
    parser.add_argument(
        "--output-onehot-csv", type=str, default=None,
        help="If provided, save one-hot encoded matrix to this CSV path"
    )
    parser.add_argument(
        "--output-transactions-json", type=str, default=None,
        help="If provided, save transactions list (JSON) to this path"
    )
    args = parser.parse_args()

    # Load and preprocess
    raw_df = load_data(args.input_csv)
    one_hot = preprocess_for_apriori(raw_df, group=args.filter_group)

    # Save one-hot encoding if requested
    if args.output_onehot_csv:
        one_hot.to_csv(args.output_onehot_csv, index=False)
        print(f"Saved one-hot encoded data to {args.output_onehot_csv}")

    # Generate transaction list and save if requested
    transactions = get_transactions(one_hot)
    if args.output_transactions_json:
        with open(args.output_transactions_json, 'w') as f:
            json.dump(transactions, f)
        print(f"Saved {len(transactions)} transactions to {args.output_transactions_json}")

    print("Preprocessing complete.")
