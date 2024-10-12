Midterm Project Report 

NAME: Pradeep Varma Ganapathiraju 
UCID: pg552 
Email Address: pg552@njit.edu 

 Date: 10/03/2024 
 Professor: Yasser Abduallah 
 Course: CS 634101 Data Mining 
                                                                     Implementation and Code Usage 

Abstract: 

In this project, I implemented and compared three algorithms for association rule mining: Brute Force, Apriori, and FP-Growth. Each algorithm was applied to transactional datasets to discover frequent itemsets and association rules, and the results were evaluated based on performance and accuracy. 

 

Introduction: 

Data mining is essential for extracting valuable patterns and insights from large datasets. One key technique, association rule mining, uncovers relationships in transactional data, frequently used in market basket analysis and recommendation systems. In this project, I implemented three algorithms: Brute Force, Apriori, and FP-Growth, to identify frequent itemsets and generate association rules. The Brute Force method explores all item combinations but is computationally intensive. The Apriori Algorithm optimizes this by using the property that subsets of frequent itemsets must also be frequent. Lastly, the FP-Growth Algorithm uses an FP-Tree to efficiently mine frequent itemsets without generating candidate sets. By applying these methods to transactional datasets from sources like Amazon and Costco, I compared their performance in terms of execution time, accuracy, and scalability.  

In this implementation, I applied Brute Force, Apriori, and FP-Growth algorithms to custom transactional datasets from retail stores, such as Amazon, Best Buy, Costco, Nike, and Walgreens. The objective was to discover frequent itemsets and generate association rules. Key steps in this process included: 

I created five custom datasets using the data from kaggle and enhanced it with additional data using ChatGPT, ensuring that each dataset contained at least 1000+ transactions. 

I loaded the datasets from CSV files into Jupyter Notebook and implemented the necessary algorithms. 

I processed the datasets so that each row represented a transaction, and each column represented an item. I collected user input during runtime for setting support and confidence thresholds. 

I implemented the Brute Force method to generate frequent itemsets and derive association rules manually. 

I used the Apriori Algorithm and FP-Growth Algorithm to compare its results with those from the brute force method to evaluate performance and accuracy. 

 

Core Concepts and Principles: 

Frequent Itemset Discovery: All three algorithms aim to discover frequent itemsets, revealing customer purchasing patterns. 

Support and Confidence: These are important measures in the analysis. Support tells us how often an item or group of items appears in the transactions, while confidence shows how likely it is that items are bought together. 

Association Rules: The rules generated from frequent itemsets help identify items frequently bought together, offering insights for optimizing marketing strategies. 

Project Workflow: 

This project follows a step-by-step process using the Brute Force, Apriori, and FP-Growth algorithms. 

Data Loading and Preprocessing: We start by loading transaction data from stores like Amazon, Best Buy, and Costco. Each transaction contains a list of items purchased by a customer. To ensure the data is clean and ready for analysis, we remove duplicate items and sort them in a specific order. 

Setting Minimum Support and Confidence: The user provides the minimum support and confidence levels. These values help us focus on frequent itemsets and strong associations, filtering out less important patterns. 

Generating Candidate Itemsets: Using the Apriori Algorithm, we generate combinations of items, starting with single items and increasing to pairs, triplets, and so on. This process considers only combinations that meet the minimum support requirement to save time and resources. 

Support Calculation: For each combination of items (candidate itemset), we count how many transactions contain that combination. If an itemset appears frequently enough (meets the minimum support), we keep it; otherwise, we discard it. 

Confidence Calculation: After identifying frequent itemsets, we calculate how strong the association is between items. This is done by checking how often one item is bought when another item is also bought (confidence). 

Creating Association Rules: Finally, we generate association rules from the frequent itemsets that meet both the minimum support and confidence thresholds. These rules show which items are often bought together, helping businesses with marketing strategies and product placement. 

 

Conclusion: 

In conclusion, the brute force method is inefficient for large datasets, as it becomes computationally expensive. Both Apriori and FP-Growth are more suitable for large datasets, with FP-Growth being the fastest due to its use of the FP-tree. This project demonstrated the effectiveness of Apriori and FP-Growth in generating frequent itemsets and association rules efficiently.  
