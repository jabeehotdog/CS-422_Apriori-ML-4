from itertools import combinations

def load_data():
    """Load TV show data and convert to transaction format"""
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '../../data/Tv_Shows_modified.csv')
    transactions = []
    
    try:
        with open(file_path, 'r') as f:
            # Read the first line to get headers (show names)
            headers = f.readline().strip().split(',')
            
            # Process all remaining lines
            for line in f:
                if not line.strip():
                    continue
                    
                # Split the line and create transaction
                shows = line.strip().split(',')
                # Filter out empty strings
                shows = [show for show in shows if show.strip()]
                if shows:
                    transactions.append(shows)
                    
    except Exception as e:
        print(f"Error loading data: {e}")
    
    return transactions

def generate_frequent_itemsets(transactions, min_support):
    """Generate frequent itemsets using Apriori algorithm"""
    # Step 1: Generate 1-itemsets
    item_counts = {}
    total_transactions = len(transactions)
    
    # Count each item's frequency
    for transaction in transactions:
        for item in transaction:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
    
    # Filter by minimum support
    frequent_1_itemsets = {item: count/total_transactions for item, count in item_counts.items() 
                          if count/total_transactions >= min_support}
    
    # Initialize iteration tracking
    all_frequent_itemsets = {1: frequent_1_itemsets}
    k = 2
    
    print("\nApriori Iterations:")
    print(f"Iteration 1 (1-itemsets): Found {len(frequent_1_itemsets)} frequent items")
    
    while True:
        # Generate candidate itemsets of size k
        candidates = set()
        current_frequent = list(all_frequent_itemsets[k-1].keys())
        
        # Generate combinations of size k
        for i in range(len(current_frequent)):
            for j in range(i+1, len(current_frequent)):
                itemset1 = current_frequent[i]
                itemset2 = current_frequent[j]
                
                if isinstance(itemset1, str):
                    itemset1 = (itemset1,)
                if isinstance(itemset2, str):
                    itemset2 = (itemset2,)
                
                candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                if len(candidate) == k:
                    candidates.add(candidate)
        
        # Count support for each candidate
        candidate_support = {}
        for candidate in candidates:
            count = sum(1 for transaction in transactions 
                       if all(item in transaction for item in candidate))
            support = count/total_transactions
            if support >= min_support:
                candidate_support[candidate] = support
        
        # Check if we found any frequent itemsets
        if not candidate_support:
            print(f"\nIteration {k}: No frequent itemsets found. Stopping.")
            break
            
        # Store the results for this iteration
        all_frequent_itemsets[k] = candidate_support
        print(f"\nIteration {k} ({k}-itemsets): Found {len(candidate_support)} frequent itemsets")
        
        k += 1
    
    # Combine all itemsets into a single dictionary
    final_frequent_itemsets = {}
    for size, itemsets in all_frequent_itemsets.items():
        for itemset, support in itemsets.items():
            final_frequent_itemsets[itemset] = support
    
    return final_frequent_itemsets

def generate_rules(frequent_itemsets, transactions, min_confidence):
    """
    Generate association rules ONLY from the frequent itemsets of the last iteration.
    """
    print("\nGenerating association rules (last iteration only)...")
    rules = []

    # Find the size of the largest frequent itemsets
    max_size = 0
    for itemset in frequent_itemsets:
        if isinstance(itemset, tuple):
            max_size = max(max_size, len(itemset))
        elif isinstance(itemset, str):
            max_size = max(max_size, 1)

    print(f"Size of the largest frequent itemsets: {max_size}")

    # Iterate through only the largest frequent itemsets
    last_iteration_itemsets = {
        itemset: support
        for itemset, support in frequent_itemsets.items()
        if (isinstance(itemset, tuple) and len(itemset) == max_size) or (isinstance(itemset, str) and max_size == 1)
    }
    print(f"Number of largest frequent itemsets: {len(last_iteration_itemsets)}")

    for itemset, support in last_iteration_itemsets.items():
        if isinstance(itemset, tuple) and len(itemset) > 1:
            # Generate rules for each possible antecedent size
            for antecedent_size in range(1, len(itemset)):
                # Get all combinations of antecedent_size items
                for antecedent_tuple in combinations(itemset, antecedent_size):
                    antecedent = tuple(sorted(antecedent_tuple)) if antecedent_size > 1 else antecedent_tuple[0]
                    # Get the remaining items as consequent
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    
                    # Make sure we have a valid antecedent and consequent
                    if antecedent and consequent:
                        # Get support for the antecedent
                        # For 1-item antecedents, we need to check both tuple and string forms
                        if isinstance(antecedent, tuple):
                            antecedent_support = frequent_itemsets.get(antecedent, 0)
                        else:
                            antecedent_support = frequent_itemsets.get(antecedent, 0)
                        
                        if antecedent_support > 0:
                            confidence = support / antecedent_support
                            # Generate all rules regardless of confidence threshold
                            print(f"Rule found: {antecedent} -> {consequent}, Confidence: {confidence:.2f}, Support: {support:.4f}")
                            rules.append((antecedent, consequent, support, confidence))
        elif isinstance(itemset, str) and max_size > 1:
            # If the largest itemset size is > 1, we won't generate rules from 1-itemsets in the last iteration
            pass
        elif isinstance(itemset, str) and max_size == 1:
            # If the last iteration only has 1-itemsets, no rules with antecedent and consequent can be formed
            print("Note: The last iteration only contains 1-itemsets, so no association rules can be generated.")

    # Sort rules by confidence
    rules.sort(key=lambda x: x[3], reverse=True)
    print(f"\nGenerated {len(rules)} rules (from last iteration). Showing all rules regardless of confidence threshold.")
    return rules