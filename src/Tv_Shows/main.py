import os
import sys

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from custom_apriori import load_data, generate_frequent_itemsets, generate_rules
from visualization import Visualizer

def main():
    # Load data
    transactions = load_data()
    
    # Set parameters (more lenient for better results)
    min_support = 0.03 
    min_confidence = 0.6
    
    # Generate frequent itemsets
    frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
    print(f"\nFound {len(frequent_itemsets)} frequent itemsets")
    
    # Generate rules
    rules = generate_rules(frequent_itemsets, transactions, min_confidence)
    print(f"\nGenerated {len(rules)} rules")
    
    try:
        # Try to visualize results
        print("\nGenerating visualizations...")
        visualizer = Visualizer()
        
        # Display results
        total_transactions = len(transactions)
        visualizer.display_frequent_itemsets(frequent_itemsets, total_transactions, top_n=20)
        visualizer.display_rules(rules, min_confidence)
    except ImportError as e:
        print(f"\nVisualization failed (Error: {str(e)}). Displaying results in text format:")
        
        # Fallback to text display
        print("\nFrequent Itemsets:")
        sorted_items = sorted(frequent_itemsets.items(), key=lambda x: x[1], reverse=True)
        for itemset, support in sorted_items[:10]:
            print(f"{itemset}: Support = {support:.2f}")
        
        print("\nAssociation Rules:")
        sorted_rules = sorted(rules, key=lambda x: x[2], reverse=True)
        for antecedent, consequent, confidence in sorted_rules[:10]:
            print(f"{' & '.join(antecedent)} -> {' & '.join(consequent)}: Confidence = {confidence:.2f}")

if __name__ == "__main__":
    main()
