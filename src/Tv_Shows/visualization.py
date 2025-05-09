import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class Visualizer:
    def display_rules(self, rules, min_confidence):
        """
        Display association rules using a bar chart.
        """
        if not rules:
            print("\nNo association rules found. Try lowering the minimum confidence threshold.")
            return
            
        # Separate rules by confidence threshold
        successful_rules = []
        failed_rules = []
        
        for antecedent, consequent, support, confidence in rules:
            # Format the rule as a string for display
            antecedent_str = ' & '.join(antecedent) if isinstance(antecedent, tuple) else antecedent
            consequent_str = ' & '.join(consequent) if isinstance(consequent, tuple) else consequent
            rule_str = f"{antecedent_str} -> {consequent_str}"
            
            if confidence >= min_confidence:
                successful_rules.append((rule_str, confidence))
            else:
                failed_rules.append((rule_str, confidence))
        
        # Calculate figure size based on number of rules
        num_rules = len(rules)
        
        # Display successful rules in a separate window
        if successful_rules:
            successful_labels, successful_confidences = zip(*successful_rules)
            
            fig1, ax1 = plt.subplots(figsize=(15, max(6, len(successful_rules) * 0.2)))
            y_pos1 = range(len(successful_labels))
            bars1 = ax1.barh(y_pos1, successful_confidences, color='skyblue')
            
            ax1.set_yticks(y_pos1)
            ax1.set_yticklabels(successful_labels, fontsize=8) 
            ax1.set_xlabel('Confidence')
            ax1.set_title(f'Successful Rules (Confidence ≥ {min_confidence:.2f})')
            
            # Add confidence values as text
            for bar in bars1:
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.2f}',
                        ha='left', va='center')
            
            plt.tight_layout()
            plt.show()
        
        # Display failed rules in a separate window
        if failed_rules:
            failed_labels, failed_confidences = zip(*failed_rules)
            
            fig2, ax2 = plt.subplots(figsize=(15, max(6, len(failed_rules) * 0.2)))
            y_pos2 = range(len(failed_labels))
            bars2 = ax2.barh(y_pos2, failed_confidences, color='salmon')
            
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(failed_labels, fontsize=8)
            ax2.set_xlabel('Confidence')
            ax2.set_title(f'Failed Rules (Confidence < {min_confidence:.2f})')
            
            # Add confidence values as text
            for bar in bars2:
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.2f}',
                        ha='left', va='center')
            
            plt.tight_layout()
            plt.show()

    def __init__(self):
        plt.style.use('seaborn')  # Use seaborn style for better aesthetics

    def __repr__(self):
        return "Visualizer()"

    def display_frequent_itemsets(self, frequent_itemsets, total_transactions, top_n=20):
        """
        Display frequent itemsets in a horizontal table format, showing iterations of Apriori algorithm.
        """
        # Group itemsets by their size
        itemsets_by_size = {}
        for itemset, support in frequent_itemsets.items():
            if isinstance(itemset, tuple):
                size = len(itemset)
                itemset_str = ' & '.join(itemset)
            else:
                size = 1
                itemset_str = str(itemset)

            if size not in itemsets_by_size:
                itemsets_by_size[size] = []
            itemsets_by_size[size].append((itemset_str, support))

        # Sort each group by support
        for size in itemsets_by_size:
            itemsets_by_size[size].sort(key=lambda x: x[1], reverse=True)
            itemsets_by_size[size] = itemsets_by_size[size][:top_n]

        # Create a grid for the tables
        num_sizes = len(itemsets_by_size)

        base_width = 12  # Base width for 1-2 tables
        width_per_table = 3  # Additional width per extra table
        fig_width = base_width + max(0, num_sizes - 2) * width_per_table
        fig = plt.figure(figsize=(fig_width, 6))
        if num_sizes == 1:
            wspace = 0.1
        elif num_sizes == 2:
            wspace = 0.15
        else:
            wspace = 0.15

        left_margin = 0.01  
        right_margin = 0.98 

        if num_sizes > 1:
            right_margin = 0.975

        grid = plt.GridSpec(1, num_sizes, wspace=wspace, hspace=0.1,
                            left=left_margin, right=right_margin, top=0.85, bottom=0.1)

        # Find the maximum number of rows across all tables
        max_rows = max(len(itemsets) for itemsets in itemsets_by_size.values())

        # Display each size
        for i, (size, itemsets) in enumerate(sorted(itemsets_by_size.items())):
            ax = plt.subplot(grid[0, i])
            ax.axis('off')

            # Create data for this size
            data = [[f'Itemset (Size {size})', 'Support']]
            for itemset_str, support in itemsets:
                # Calculate actual count from support value
                count = round(support * total_transactions)
                data.append([itemset_str, str(count)])

            # Create table
            table = plt.table(cellText=data,
                                loc='center',
                                cellLoc='center')

            # Adjust font sizes and colors
            for key, cell in table.get_celld().items():
                cell.set_height(0.4)  # Increased row height
                if key[0] == 0:  # Header row
                    cell.set_facecolor('#f0f0f0')
                    cell.set_fontsize(10)  # Reduced header font size
                else:
                    cell.set_facecolor('white')
                    cell.set_fontsize(9)  # Reduced regular font size

            # Adjust column widths
            table.auto_set_column_width(col=list(range(len(data[0]))))

            # Set specific widths for columns
            for key, cell in table.get_celld().items():
                if key[1] == 0:  # Itemset column
                    cell.set_width(0.7)
                else:  # Support column
                    cell.set_width(0.3)

            # Make rows compact
            for key, cell in table.get_celld().items():
                cell.set_height(0.8 / (max_rows + 1))

        plt.suptitle('Frequent Itemsets by Size', fontsize=15, y=0.95)

        plt.show()