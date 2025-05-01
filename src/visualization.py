import matplotlib.pyplot as plt


def plot_itemset_support(frequent_itemsets, top_n=10, show=True):
    """
    Plot the support of the top_n frequent itemsets as a bar chart.
    """
    # Prepare labels and values
    pairs = [(tuple(sorted(list(itemset))), supp) for itemset, supp in frequent_itemsets.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_n]
    labels = [", ".join(p[0]) for p in top]
    values = [p[1] for p in top]

    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.ylabel('Support')
    plt.title(f'Top {top_n} Frequent Itemsets')
    plt.tight_layout()
    if show:
        plt.show()


def plot_rules(rules, metric='confidence', top_n=10, show=True):
    """
    Plot the top_n association rules by a given metric (confidence or lift).
    """
    # Sort rules
    sorted_rules = sorted(rules, key=lambda r: r.get(metric, 0), reverse=True)[:top_n]
    labels = [f"{', '.join(sorted(r['antecedent']))}→{', '.join(sorted(r['consequent']))}" for r in sorted_rules]
    values = [r.get(metric, 0) for r in sorted_rules]

    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.ylabel(metric.capitalize())
    plt.title(f'Top {top_n} Rules by {metric.capitalize()}')
    plt.tight_layout()
    if show:
        plt.show()
