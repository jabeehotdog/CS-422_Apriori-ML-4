import itertools


def generate_candidates(prev_frequents, k):
    """
    Generate k-item candidate sets from (k-1)-item frequent itemsets.
    """
    candidates = set()
    items = list(prev_frequents)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            union = items[i] | items[j]
            if len(union) == k:
                # prune: all (k-1)-subsets must be frequent
                subsets = itertools.combinations(union, k - 1)
                if all(frozenset(sub) in prev_frequents for sub in subsets):
                    candidates.add(union)
    return candidates


def get_frequent_itemsets(transactions, min_support):
    """
    Identify all frequent itemsets using the Apriori principle.
    Returns a dict mapping frozenset(itemset) to support (float).
    """
    transaction_list = list(transactions)
    total = len(transaction_list)

    # Count support for single items (L1)
    counts = {}
    for t in transaction_list:
        for item in t:
            counts[item] = counts.get(item, 0) + 1
    L1 = {frozenset([item]): count / total for item, count in counts.items() if count / total >= min_support}

    frequents = dict(L1)
    k = 2
    current_frequents = set(L1.keys())

    # Iteratively build larger frequent itemsets
    while current_frequents:
        candidates = generate_candidates(current_frequents, k)
        candidate_counts = {}
        for t in transaction_list:
            tset = set(t)
            for c in candidates:
                if c.issubset(tset):
                    candidate_counts[c] = candidate_counts.get(c, 0) + 1
        Lk = {c: cnt / total for c, cnt in candidate_counts.items() if cnt / total >= min_support}
        current_frequents = set(Lk.keys())
        frequents.update(Lk)
        k += 1

    return frequents


def generate_rules(frequent_itemsets, min_confidence):
    """
    Generate association rules from frequent itemsets.
    Returns a list of dicts with keys: antecedent, consequent, support, confidence, lift.
    """
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        # all possible non-empty antecedents
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                ant = frozenset(antecedent)
                cons = itemset - ant
                ant_support = frequent_itemsets.get(ant, 0)
                if ant_support == 0:
                    continue
                confidence = support / ant_support
                cons_support = frequent_itemsets.get(cons, 0)
                lift = confidence / cons_support if cons_support > 0 else 0
                if confidence >= min_confidence:
                    rules.append({
                        'antecedent': ant,
                        'consequent': cons,
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
    return rules


def apriori(transactions, min_support=0.05, min_confidence=0.5):
    """
    Run Apriori over the given transactions.
    Returns a tuple (frequent_itemsets, rules).
    - frequent_itemsets: dict[frozenset -> support]
    - rules: list of rule dicts
    """
    frequents = get_frequent_itemsets(transactions, min_support)
    rules = generate_rules(frequents, min_confidence)
    return frequents, rules
