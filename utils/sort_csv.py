# importing pandas package
import pandas as pd

# Set option to display all columns
pd.set_option('display.max_columns', None)
# assign dataset
csvData = pd.read_csv("pls_feature_assoc_stratified.csv")


# sort data frame
csvData.sort_values(["best_f1"], 
                    axis=0,
                    ascending=[False], 
                    inplace=True)

# displaying sorted data frame

print("\nAfter sorting:")
print(csvData.head(50))