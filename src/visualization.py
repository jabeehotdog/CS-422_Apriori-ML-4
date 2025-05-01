import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        pass  # No initialization needed for now, but keeps it extensible

    def plot_supports(self, support_data, top_n=10):
        """
        Plot the top N frequent itemsets by support.
        """
        sorted_items = sorted(support_data.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]
        labels = [' & '.join(item[0]) for item in top_items]
        values = [item[1] for item in top_items]

        plt.figure(figsize=(10, 6))
        plt.barh(labels, values)
        plt.xlabel('Support')
        plt.title(f'Top {top_n} Frequent Itemsets by Support')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_confidences(self, rules, top_n=10):
        """
        Plot the top N rules by confidence.
        """
        sorted_rules = sorted(rules, key=lambda x: x[2], reverse=True)
        top_rules = sorted_rules[:top_n]
        labels = [f"{' & '.join(a)} → {' & '.join(c)}" for a, c, _ in top_rules]
        values = [conf for _, _, conf in top_rules]

        plt.figure(figsize=(10, 6))
        plt.barh(labels, values)
        plt.xlabel('Confidence')
        plt.title(f'Top {top_n} Association Rules by Confidence')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
