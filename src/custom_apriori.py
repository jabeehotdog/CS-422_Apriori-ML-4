class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.support_data = {}
        self.frequent_itemsets = []
        self.rules = []

    # Helper function to generate combinations
    def _generate_combinations(self, items, k):
        def combine(start, path):
            if len(path) == k:
                combinations.append(path)
                return
            for i in range(start, len(items)):
                combine(i + 1, path + [items[i]])

        combinations = []
        combine(0, [])
        return combinations

    def fit(self, transactions):
        self.frequent_itemsets = []
        self.support_data = {}
        num_transactions = len(transactions)

        # Step 1: 1-itemsets
        item_count = {}
        for transaction in transactions:
            for item in transaction:
                key = tuple([item])
                item_count[key] = item_count.get(key, 0) + 1

        current_l_set = []
        for item, count in item_count.items():
            support = count / num_transactions
            if support >= self.min_support:
                current_l_set.append(list(item))
                self.support_data[tuple(sorted(item))] = support

        k = 2
        while current_l_set:
            self.frequent_itemsets.extend(current_l_set)
            unique_items = sorted(set(i for items in current_l_set for i in items))
            candidates = self._generate_combinations(unique_items, k)

            item_count = {}
            for transaction in transactions:
                for candidate in candidates:
                    if all(item in transaction for item in candidate):
                        key = tuple(sorted(candidate))
                        item_count[key] = item_count.get(key, 0) + 1

            current_l_set = []
            for item, count in item_count.items():
                support = count / num_transactions
                if support >= self.min_support:
                    current_l_set.append(list(item))
                    self.support_data[item] = support
            k += 1

    def generate_rules(self):
        self.rules = []
        for itemset in self.frequent_itemsets:
            if len(itemset) >= 2:
                n = len(itemset)
                for i in range(1, n):
                    subsets = self._generate_combinations(itemset, i)
                    for antecedent in subsets:
                        consequent = [i for i in itemset if i not in antecedent]
                        if consequent:
                            ant_key = tuple(sorted(antecedent))
                            full_key = tuple(sorted(itemset))
                            confidence = self.support_data[full_key] / self.support_data[ant_key]
                            if confidence >= self.min_confidence:
                                self.rules.append((antecedent, consequent, confidence))

    def print_frequent_itemsets(self):
        print("Frequent Itemsets:")
        for item in self.frequent_itemsets:
            print(item, "Support:", round(self.support_data[tuple(sorted(item))], 4))

    def print_rules(self):
        print("\nAssociation Rules:")
        for antecedent, consequent, confidence in self.rules:
            print(f"{antecedent} => {consequent}, Confidence: {round(confidence, 2)}")
