import argparse
import json
from src.custom_apriori import Apriori
from src.preprocess import Preprocessor
from src.visualization import Visualizer

def main():
    parser = argparse.ArgumentParser(
        description="Apriori on Interest Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-csv", 
        type=str, 
        default="./data/kaggle_Interests_group.csv",
        help="Path to input CSV"
    )
    parser.add_argument("--filter-group", type=str, default=None, help="Filter by group (C, P, R, etc.)")
    parser.add_argument("--min-support", type=float, default=0.7, help="Minimum support")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence")
    parser.add_argument("--output-onehot-csv", type=str, default=None, help="Path to save one-hot CSV")
    parser.add_argument("--output-transactions-json", type=str, default=None, help="Path to save transactions")

    args = parser.parse_args()

    # Initialize and preprocess
    preprocessor = Preprocessor(args.input_csv)
    one_hot = preprocessor.preprocess(group=args.filter_group)

    if args.output_onehot_csv:
        one_hot.to_csv(args.output_onehot_csv, index=False)
        print(f"Saved one-hot encoded data to {args.output_onehot_csv}")

    transactions = preprocessor.get_transactions(one_hot)

    if args.output_transactions_json:
        with open(args.output_transactions_json, 'w') as f:
            json.dump(transactions, f)
        print(f"Saved {len(transactions)} transactions to {args.output_transactions_json}")

    # Run Apriori algorithm
    model = Apriori(min_support=args.min_support, min_confidence=args.min_confidence)
    model.fit(transactions)
    model.generate_rules()
    model.print_frequent_itemsets()
    model.print_rules()

    viz = Visualizer()
    viz.plot_supports(model.support_data, top_n=10)
    viz.plot_confidences(model.rules, top_n=10)

if __name__ == "__main__":
    main()

