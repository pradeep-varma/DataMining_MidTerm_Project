#!/usr/bin/env python
# coding: utf-8

# ## Brute Force Approach for Transaction Analysis

# In[21]:


import pandas as pd
import itertools
import time
from tabulate import tabulate

def load_csv_as_transactions(csv_path):
    df = pd.read_csv(csv_path, usecols=[1], skiprows=1, names=['items'])
    return df['items'].apply(lambda x: set(x.split(' , '))).tolist()

def calculate_support(itemset, transactions):
    return sum(set(itemset).issubset(t) for t in transactions)

def find_frequent_sets(transactions, min_support, max_length=3):
    all_items = {item for transaction in transactions for item in transaction}
    subsets = []
    for length in range(1, max_length + 1):
        for combo in itertools.combinations(all_items, length):
            if (support := calculate_support(combo, transactions)) >= min_support:
                subsets.append((set(combo), support))
    return subsets

def derive_rules(frequent_sets, transactions, min_confidence):
    rules = []
    for base, base_support in frequent_sets:
        for subset in map(set, itertools.chain.from_iterable(itertools.combinations(base, r) for r in range(1, len(base)))):
            consequent = base - subset
            if len(consequent) == 0:
                continue
            subset_support = calculate_support(subset, transactions)
            confidence = base_support / subset_support if subset_support else 0
            if confidence >= min_confidence:
                rules.append((subset, consequent, confidence))
    return rules

def process_and_analyze(file_path, min_support=2, min_confidence=0.5):
    transactions = load_csv_as_transactions(file_path)
    frequent_itemsets = find_frequent_sets(transactions, min_support)
    rules = derive_rules(frequent_itemsets, transactions, min_confidence)
    return frequent_itemsets, rules

def format_output(itemsets, rules):
    itemsets_table = tabulate(
        [(i + 1, ', '.join(itemset), support) for i, (itemset, support) in enumerate(itemsets)],
        headers=["No.", "Frequent Itemset", "Support"],
        tablefmt="pretty"
    )

    rules_table = tabulate(
        [(i + 1, ', '.join(subset), ', '.join(consequent), f"{confidence:.2f}") for i, (subset, consequent, confidence) in enumerate(rules)],
        headers=["No.", "Antecedent", "Consequent", "Confidence"],
        tablefmt="pretty"
    )

    return itemsets_table, rules_table

start_time = time.time()

file_paths = [
    "/Users/pradeepvarma/Downloads/dataset/walmart_data.csv",
    "/Users/pradeepvarma/Downloads/dataset/amazon_data.csv",
    "/Users/pradeepvarma/Downloads/dataset/best_buy_data.csv",
    "/Users/pradeepvarma/Downloads/dataset/costco_data.csv",
    "/Users/pradeepvarma/Downloads/dataset/walgreens_data.csv"
]

for path in file_paths:
    itemsets, association_rules = process_and_analyze(path, 0.05, 0.5)
    print(f"Results for {path}:")
    
    # Format the output
    itemsets_table, rules_table = format_output(itemsets, association_rules)
    
    print("\nFrequent Itemsets:")
    print(itemsets_table)
    print("\nAssociation Rules:")
    print(rules_table)
    print("\n")

print("Execution Time:", time.time() - start_time)


# ## Apriori Algorithm for Transaction Analysis
# 

# In[22]:


import pandas as pd
from time import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from tabulate import tabulate

def read_transactions(csv_file):
    transaction_data = pd.read_csv(csv_file)['Filtered Transaction'].str.split(', ').tolist()
    return transaction_data

def perform_analysis(dataset_path, support_threshold, confidence_level):
    tick = time() 
    
    transactions = read_transactions(dataset_path)
    
    # Convert transactions into a one-hot encoded DataFrame
    encoder = TransactionEncoder()
    transaction_matrix = encoder.fit_transform(transactions)
    transaction_df = pd.DataFrame(transaction_matrix, columns=encoder.columns_)
    
    # Generate frequent itemsets using Apriori
    frequent_sets = apriori(transaction_df, min_support=support_threshold, use_colnames=True)
    
    # Derive association rules from the frequent itemsets
    derived_rules = association_rules(frequent_sets, metric="confidence", min_threshold=confidence_level)
    
    dataset_name = dataset_path.split('/')[-1]  # Extracting filename for display
    
    print(f"\nTransactions from {dataset_name}:\n")
    transaction_table = tabulate(transactions, headers=["Transaction Items"], tablefmt="pretty")
    print(transaction_table)
    
    print("\nDerived Association Rules:\n")
    if not derived_rules.empty:
        rules_table = tabulate(
            derived_rules[['antecedents', 'consequents', 'support', 'confidence']].values,
            headers=['Antecedents', 'Consequents', 'Support', 'Confidence'],
            tablefmt='pretty'
        )
        print(rules_table)
    else:
        print("No association rules found for the given parameters.")
    
    tock = time()
    print(f"\nProcessing time for {dataset_name}: {tock - tick:.2f} seconds\n")

def initiate_analysis():
    datasets = [
        "/Users/pradeepvarma/Downloads/dataset/walmart_data.csv",
        "/Users/pradeepvarma/Downloads/dataset/amazon_data.csv",
        "/Users/pradeepvarma/Downloads/dataset/best_buy_data.csv",
        "/Users/pradeepvarma/Downloads/dataset/costco_data.csv",
        "/Users/pradeepvarma/Downloads/dataset/walgreens_data.csv"
    ]
    
    min_support = float(input("Enter minimum support value (e.g., 0.05 for 5%): "))
    min_confidence = float(input("Enter minimum confidence value (e.g., 0.5 for 50%): "))

    for path in datasets:
        perform_analysis(path, min_support, min_confidence)

if __name__ == "__main__":
    initiate_analysis()


# In[23]:


import pandas as pd
from time import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from tabulate import tabulate

def load_and_filter_tx(file, min_items=2):
    df = pd.read_csv(file)
    
    txs = df['Filtered Transaction'].apply(lambda x: x.split(', '))
    filtered_txs = [tx for tx in txs if len(tx) >= min_items]
    return filtered_txs

def analyze_data(file, min_sup, min_conf):
    start = time()  # Start timing

    txs = load_and_filter_tx(file)
    
    enc = TransactionEncoder()
    tx_array = enc.fit(txs).transform(txs)
    tx_df = pd.DataFrame(tx_array, columns=enc.columns_)
    
    freq_sets = apriori(tx_df, min_support=min_sup, use_colnames=True)
    ass_rules = association_rules(freq_sets, metric="confidence", min_threshold=min_conf)
    
    print(f"\nFiltered Transactions for {file.split('/')[-1]}:\n")
    tx_table = tabulate(txs, headers=["Transaction Items"], tablefmt="pretty")
    print(tx_table)
    
    print("\nGenerated Association Rules:\n")
    if not ass_rules.empty:
        rules_table = tabulate(
            ass_rules[['antecedents', 'consequents', 'support', 'confidence']].values,
            headers=['Antecedents', 'Consequents', 'Support', 'Confidence'],
            tablefmt='pretty'
        )
        print(rules_table)
    else:
        print("No association rules found for the given parameters.")
    
    print(f"\nTime taken for {file.split('/')[-1]}: {time() - start:.2f} seconds\n")

def run_analysis():
    datasets = {
        1: '/Users/pradeepvarma/Downloads/dataset/walmart_data.csv',
        2: '/Users/pradeepvarma/Downloads/dataset/amazon_data.csv',
        3: '/Users/pradeepvarma/Downloads/dataset/best_buy_data.csv',
        4: '/Users/pradeepvarma/Downloads/dataset/costco_data.csv',
        5: '/Users/pradeepvarma/Downloads/dataset/walgreens_data.csv',
    }
    
    print("Select dataset(s):")
    for k, v in datasets.items():
        print(f"{k} - {v.split('/')[-1]}")
    
    choices = input("Choices (e.g., 1,3): ").split(',')
    min_sup = float(input("Minimum support (e.g., 0.05): "))
    min_conf = float(input("Minimum confidence (e.g., 0.5): "))

    for choice in choices:
        analyze_data(datasets[int(choice)], min_sup, min_conf)

if __name__ == "__main__":
    run_analysis()


# ## FP-Growth Algorithm for Transaction Analysis

# In[25]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from tabulate import tabulate

def fetch_transactions(csv_path):
    data = pd.read_csv(csv_path)['Filtered Transaction'].str.split(', ').tolist()
    return data

def fp_growth_analysis(csv_path, support_level, confidence_level):
    transactions = fetch_transactions(csv_path)
    
    # Encode the transactions into a one-hot encoded DataFrame
    encoder = TransactionEncoder()
    encoded_data = encoder.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)
    
    # Generate frequent itemsets using FP-Growth
    itemsets = fpgrowth(df_encoded, min_support=support_level, use_colnames=True)
    
    # Derive association rules from the frequent itemsets
    rules = association_rules(itemsets, metric="confidence", min_threshold=confidence_level)
    
    print(f"\nFrequent Itemsets from {csv_path.split('/')[-1]}:\n")
    if not itemsets.empty:
        itemsets_table = tabulate(
            itemsets[['itemsets', 'support']].values,
            headers=['Itemsets', 'Support'],
            tablefmt='pretty'
        )
        print(itemsets_table)
    else:
        print("No frequent itemsets found for the given parameters.")
    
    print("\nDerived Association Rules:\n")
    if not rules.empty:
        rules_table = tabulate(
            rules[['antecedents', 'consequents', 'support', 'confidence']].values,
            headers=['Antecedents', 'Consequents', 'Support', 'Confidence'],
            tablefmt='pretty'
        )
        print(rules_table)
    else:
        print("No association rules found for the given parameters.")
    print("\n")

def run_analysis():
    dataset_paths = [        
        "/Users/pradeepvarma/Downloads/dataset/walmart_data.csv",
        "/Users/pradeepvarma/Downloads/dataset/amazon_data.csv",
        "/Users/pradeepvarma/Downloads/dataset/best_buy_data.csv",
        "/Users/pradeepvarma/Downloads/dataset/costco_data.csv",
        "/Users/pradeepvarma/Downloads/dataset/walgreens_data.csv"
    ]
    
    min_support = float(input("Enter minimum support value (e.g., 0.05 for 5%): "))
    min_confidence = float(input("Enter minimum confidence value (e.g., 0.5 for 50%): "))
    
    for path in dataset_paths:
        fp_growth_analysis(path, min_support, min_confidence)

if __name__ == "__main__":
    run_analysis()


# In[24]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from tabulate import tabulate
import time

def get_txs(file):
    """Fetch transactions from CSV."""
    txs = pd.read_csv(file)['Filtered Transaction'].str.split(', ').tolist()
    return txs

def analyze_fp_growth(file, sup, conf):
    """Analyzes transactions using the FP-Growth method."""
    start_time = time.time()  # Start the timer

    txs = get_txs(file)
    
    # Encode transactions into a DataFrame
    enc = TransactionEncoder()
    tx_data = enc.fit(txs).transform(txs)
    df_tx = pd.DataFrame(tx_data, columns=enc.columns_)
    
    # Generate frequent itemsets using FP-Growth
    frequent_itemsets = fpgrowth(df_tx, min_support=sup, use_colnames=True)
    
    # Derive association rules from the frequent itemsets
    assoc_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=conf)
    
    # Display results in a structured table format
    print(f"\nFrequent Itemsets from {file.split('/')[-1]}:\n")
    if not frequent_itemsets.empty:
        itemsets_table = tabulate(
            frequent_itemsets[['itemsets', 'support']],
            headers=['Itemsets', 'Support'],
            tablefmt='pretty'
        )
        print(itemsets_table)
    else:
        print("No frequent itemsets found for the given parameters.")

    print("\nDerived Association Rules:\n")
    if not assoc_rules.empty:
        rules_table = tabulate(
            assoc_rules[['antecedents', 'consequents', 'support', 'confidence']],
            headers=['Antecedents', 'Consequents', 'Support', 'Confidence'],
            tablefmt='pretty'
        )
        print(rules_table)
    else:
        print("No association rules found for the given parameters.")

    end_time = time.time()  # End the timer
    print(f"\nExecution time for {file.split('/')[-1]}: {end_time - start_time:.2f} seconds\n")

def run():
    names = ['Walmart', 'Amazon', 'Best Buy', 'Costco', 'Walgreens']
    paths = {
        '1': '/Users/pradeepvarma/Downloads/dataset/walmart_data.csv',
        '2': '/Users/pradeepvarma/Downloads/dataset/amazon_data.csv',
        '3': '/Users/pradeepvarma/Downloads/dataset/best_buy_data.csv',
        '4': '/Users/pradeepvarma/Downloads/dataset/costco_data.csv',
        '5': '/Users/pradeepvarma/Downloads/dataset/walgreens_data.csv',
    }
    
    print("Choose dataset(s) to analyze:")
    for i, name in enumerate(names, start=1):
        print(f"{i} - {name}")
    choices = input("Enter dataset numbers (space-separated, e.g., 1 3): ").split()

    min_sup = float(input("\nMinimum support (e.g., 0.05): "))
    min_conf = float(input("Minimum confidence (e.g., 0.5): "))

    for choice in choices:
        if choice in paths:
            analyze_fp_growth(paths[choice], min_sup, min_conf)
        else:
            print(f"Invalid choice: {choice}")

if __name__ == "__main__":
    run()


# In[ ]:





# In[ ]:




