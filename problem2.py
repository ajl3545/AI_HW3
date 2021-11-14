# Alex Lamarche
# AI EXP HW3 : Problem 2 : problem2.py

import pandas as pd

def load_table(table,sheet):
    data = pd.read_excel(table,sheet_name=sheet)
    return data 

clean_tr_data = load_table("data/clean_data.xlsx","Sheet1")
clean_v_data = load_table("data/clean_data.xlsx","Sheet2")
clean_te_data = load_table("data/clean_data.xlsx","Sheet3")

print(clean_tr_data)