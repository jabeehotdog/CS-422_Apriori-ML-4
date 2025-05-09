import pandas as pd
from typing import List

class TVShowsPreprocessor:
    def __init__(self, path: str):
        self.path = path
        self.show_names = None
        self.transactions = None

    def load_data(self) -> None:
        """Load and preprocess TV shows data"""
        with open(self.path, 'r') as f:
            # Get show names from first line
            self.show_names = f.readline().strip().split(',')
            
            # Load transactions
            self.transactions = []
            for line in f:
                if not line.strip():
                    continue
                
                shows = line.strip().split(',')
                # Filter out empty strings
                shows = [show for show in shows if show.strip()]
                self.transactions.append(shows)

    def get_transactions(self) -> List[List[str]]:
        """Return the processed transactions"""
        if self.transactions is None:
            self.load_data()
        return self.transactions

    def get_show_names(self) -> List[str]:
        """Return list of all show names"""
        if self.show_names is None:
            self.load_data()
        return self.show_names
