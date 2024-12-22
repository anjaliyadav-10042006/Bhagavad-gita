import pandas as pd

# Load the first CSV file
file_path1 = 'Bhagwad_gita.csv'
df1 = pd.read_csv(file_path1)

# Load the second CSV file
file_path2 = 'Patanjali_yoga_sutras.csv'
df2 = pd.read_csv(file_path2)

# Convert each row to a list
chunks1 = df1.values.tolist()
chunks2 = df2.values.tolist()

# File to save the output
output_file_path = 'output.txt'

# Write each chunk to the text file with separators, using UTF-8 encoding
with open(output_file_path, 'w', encoding='utf-8') as file:
    for chunk in chunks1:
        # Convert list to string & add separator
        file.write(str(chunk) + "\n\n")
    
    # Add a separator between the contents of the two CSV files
    file.write("\n--- Content from pys.csv ---\n\n")
    
    for chunk in chunks2:
        # Convert list to string & add separator
        file.write(str(chunk) + "\n\n")
        
print(f"Data written to {output_file_path}")

