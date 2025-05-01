import argparse
import json

from src.preprocess import load_data, preprocess_for_apriori, get_transactions
from src.custom_apriori import apriori
from src.visualization import plot_itemset_support, plot_rules


def main():
    parser = argparse.ArgumentParser(description='Full Apriori pipeline')
    parser.add_argument('--input-csv', type=str, required=True,
                        help='Path to raw interest-group CSV')
    parser.add_argument('--filter-group', type=str, default=None,
                        help='Optional: only include this group')
    parser.add_argument('--min-support', type=float, default=0.05,
                        help='Minimum support threshold (fraction)')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                        help='Minimum confidence threshold (fraction)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top itemsets/rules to display')
    parser.add_argument('--output-frequents', type=str, default=None,
                        help='Save frequent itemsets (JSON) to this file')
    parser.add_argument('--output-rules', type=str, default=None,
                        help='Save association rules (JSON) to this file')
    args = parser.parse_args()

    # Load and preprocess data
    df = load_data(args.input_csv)
    onehot = preprocess_for_apriori(df, group=args.filter_group)
    transactions = get_transactions(onehot)

    # Run Apriori
    frequents, rules = apriori(transactions,
                               min_support=args.min_support,
                               min_confidence=args.min_confidence)

    # Serialize and save if requested
    if args.output_frequents:
        serial = {', '.join(sorted(list(k))): v for k, v in frequents.items()}
        with open(args.output_frequents, 'w') as f:
            json.dump(serial, f, indent=2)
    if args.output_rules:
        serial_rules = []
        for r in rules:
            serial_rules.append({
                'antecedent': list(r['antecedent']),
                'consequent': list(r['consequent']),
                'support': r['support'],
                'confidence': r['confidence'],
                'lift': r['lift']
            })
        with open(args.output_rules, 'w') as f:
            json.dump(serial_rules, f, indent=2)

    print(f"Found {len(frequents)} frequent itemsets and {len(rules)} association rules.")

    # Visualizations
    plot_itemset_support(frequents, top_n=args.top_n)
    plot_rules(rules, metric='confidence', top_n=args.top_n)


if __name__ == '__main__':
    main()
