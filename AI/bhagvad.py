import pandas as pd

load excel file
# file_path='bhagavad-gita.xlsx' 
file_path='https://raw.githubusercontent.com/anjaliyadav-10042006/Bhagavad-gita/refs/heads/main/AI/bhagavad-gita.xlsx'
df=pd.read_excel(file_path)

#convert each row to alist
chunks = df.values.tolist()

#file to save the output
output_file_path = 'output.txt'

#write each chunk to the separator on text file
with open(output_file_path,'w') as file:
    for chunk in chunks:
        #convert list to string & add separator
        file.write(str(chunk) + "\n\n")
        
print(f" Data written to {output_file_path}")        
        
