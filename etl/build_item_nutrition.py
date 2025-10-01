# etl/build_item_nutrition.py (NEW)

def build_item_nutrition(item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract nutrition values as numeric columns.
    
    Input: nutrition_nm = [["added sugar", 22], ["Calcium", 0], ...]
    Output: Flat table with numeric columns
    """
    import ast
    
    def parse_nutrition(nutrition_str):
        if pd.isna(nutrition_str):
            return {}
        try:
            nutrition_list = ast.literal_eval(nutrition_str)
            return {name.lower().strip(): value for name, value in nutrition_list}
        except:
            return {}
    
    item_df = item_df.copy()
    item_df['nutrition_dict'] = item_df['nutrition_nm'].apply(parse_nutrition)
    
    # Expand dict to columns
    nutrition_df = pd.json_normalize(item_df['nutrition_dict'])
    nutrition_df['upc_id'] = item_df['upc_nbr'].values
    
    # Rename columns to standardized names
    column_mapping = {
        'added sugar': 'added_sugar_g',
        'calcium': 'calcium_mg',
        'calories': 'calories',
        'cholesterol': 'cholesterol_mg',
        'protein': 'protein_g',
        'sodium': 'sodium_mg',
        'total fat': 'total_fat_g',
        'fiber': 'fiber_g',
        # Add more as needed
    }
    
    nutrition_df = nutrition_df.rename(columns=column_mapping)
    
    # Keep only relevant columns
    nutrient_cols = [col for col in nutrition_df.columns if col.endswith(('_g', '_mg', 'calories'))]
    nutrition_df = nutrition_df[['upc_id'] + nutrient_cols]
    
    return nutrition_df


# Usage
item_nutrition = build_item_nutrition(item_df)
item_nutrition.to_parquet('gs://your-bucket/data/item_nutrition.parquet', index=False)


# upc_id           added_sugar_g  calories  protein_g  sodium_mg  ...
# 0001234567890    22             180       12         250
# 0009876543210    5              120       15         150