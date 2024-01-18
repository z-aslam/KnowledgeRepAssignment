import pandas as pd
import re

df = pd.read_csv("./products.csv")
# df = pd.read_csv("./products_small.csv")

new_rows = []
columns = ['title', 'price']  # Initial columns for the new DataFrame



# Use str.contains to filter rows based on the pattern
filtered_df = df[df['material'].str.contains("^Main:", regex=True) == True]


# Display the resulting DataFrame
material_comp = []
name = []
price = []

for index, row in filtered_df.iterrows():
    material_section = re.findall(r'\b\d+% [A-Za-z]+\b', row['material'])
    materials = []
    name.append(row["title"])
    price.append(row["price"])
    for section in material_section[:2]:
        percent, material_name = section.split()
        materials.append(material_name)
        materials.append(float(percent[:-1])/100.0)
    material_comp.append(materials)    
        

# Create a list to store the data
data = []

# Iterate through the arrays and create dictionaries
for title, price, material_info in zip(name, price, material_comp):
    # Extract material information
    materials = [{"Material": material_info[i], "Percentage": material_info[i + 1]} for i in range(0, len(material_info), 2)]

    if len(material_info) <= 2:
        row_data = {"Title": title, "Price": price, "Material A": material_info[0], "A Percent": material_info[1], "Material B": "NaN", "B Percent": 0}
    else:
        row_data = {"Title": title, "Price": price, "Material A": material_info[0], "A Percent": material_info[1], "Material B": material_info[2], "B Percent": material_info[3]}
    
    # Create a dictionary for each row
    
    
    data.append(row_data)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data)
df.to_csv("./data.csv",index=False)
pass