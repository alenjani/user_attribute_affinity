# etl/build_attr_meta.py (UPDATED)

def build_attr_meta(item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all unique attributes from item table.
    """
    
    attributes = []
    
    # 1. Brands (same as before)
    brands = item_df[['iri_brands_nm']].drop_duplicates()
    brands = brands[brands['iri_brands_nm'].notna()]
    brands['attr_id'] = 'brand_' + brands['iri_brands_nm'].str.replace(' ', '_').str.lower()
    brands['attr_family'] = 'brand'
    brands['display_name'] = brands['iri_brands_nm']
    attributes.append(brands[['attr_id', 'attr_family', 'display_name']])
    
    # 2. Categories (same as before)
    categories = item_df[['smic_category_cd']].drop_duplicates()
    categories = categories[categories['smic_category_cd'].notna()]
    categories['attr_id'] = 'category_' + categories['smic_category_cd'].astype(str)
    categories['attr_family'] = 'category'
    categories['display_name'] = categories['smic_category_cd'].astype(str)
    attributes.append(categories[['attr_id', 'attr_family', 'display_name']])
    
    # 3. Classes (same as before)
    classes = item_df[['smic_class_cd']].drop_duplicates()
    classes = classes[classes['smic_class_cd'].notna()]
    classes['attr_id'] = 'class_' + classes['smic_class_cd'].astype(str)
    classes['attr_family'] = 'class'
    classes['display_name'] = classes['smic_class_cd'].astype(str)
    attributes.append(classes[['attr_id', 'attr_family', 'display_name']])
    
    # 4. Departments (same as before)
    depts = item_df[['department_nm']].drop_duplicates()
    depts = depts[depts['department_nm'].notna()]
    depts['attr_id'] = 'dept_' + depts['department_nm'].str.replace(' ', '_').str.lower()
    depts['attr_family'] = 'department'
    depts['display_name'] = depts['department_nm']
    attributes.append(depts[['attr_id', 'attr_family', 'display_name']])
    
    # 5. Health Claims (UPDATED - handle multi-value)
    health = item_df[['health_clain_type_dsc']].copy()
    health = health[health['health_clain_type_dsc'].notna()]
    
    # Split by "/" delimiter and explode
    health['health_clain_type_dsc'] = health['health_clain_type_dsc'].str.split('/')
    health = health.explode('health_clain_type_dsc')
    
    # Clean whitespace
    health['health_clain_type_dsc'] = health['health_clain_type_dsc'].str.strip()
    
    # Get unique health claims
    health = health[['health_clain_type_dsc']].drop_duplicates()
    health['attr_id'] = 'health_' + health['health_clain_type_dsc'].str.replace(' ', '_').str.lower()
    health['attr_family'] = 'health_claim'
    health['display_name'] = health['health_clain_type_dsc']
    attributes.append(health[['attr_id', 'attr_family', 'display_name']])
    
    # 6. Shelves (same as before)
    shelves = item_df[['shelf_nm']].drop_duplicates()
    shelves = shelves[shelves['shelf_nm'].notna()]
    shelves['attr_id'] = 'shelf_' + shelves['shelf_nm'].str.replace(' ', '_').str.lower()
    shelves['attr_family'] = 'shelf'
    shelves['display_name'] = shelves['shelf_nm']
    attributes.append(shelves[['attr_id', 'attr_family', 'display_name']])
    
    # 7. Nutrition Attributes (NEW - from nutrition_nm)
    # We'll create categorical nutrition attributes
    # Example: "high_added_sugar", "low_calories", etc.
    nutrition_attrs = extract_nutrition_attributes(item_df)
    attributes.append(nutrition_attrs[['attr_id', 'attr_family', 'display_name']])
    
    # Combine all
    attr_meta = pd.concat(attributes, ignore_index=True)
    
    # Add optional fields
    attr_meta['definition_json'] = None
    
    return attr_meta


def extract_nutrition_attributes(item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert nutrition_nm (nested list) into categorical attributes.
    
    Example input: [["added sugar", 22], ["Calcium", 0], ["Calories", 180], ...]
    
    Strategy: Create bins for key nutrients
      - added_sugar_high (>20g)
      - added_sugar_low (<5g)
      - calories_high (>200)
      - protein_high (>15g)
      etc.
    """
    import ast
    
    # Parse nutrition_nm from string representation to actual list
    def parse_nutrition(nutrition_str):
        if pd.isna(nutrition_str):
            return []
        try:
            return ast.literal_eval(nutrition_str)
        except:
            return []
    
    item_df['nutrition_parsed'] = item_df['nutrition_nm'].apply(parse_nutrition)
    
    # Convert to dict for easier access
    def to_dict(nutrition_list):
        return {name.lower().strip(): value for name, value in nutrition_list}
    
    item_df['nutrition_dict'] = item_df['nutrition_parsed'].apply(to_dict)
    
    # Extract key nutrients
    def get_nutrient(nutrition_dict, nutrient_name):
        return nutrition_dict.get(nutrient_name, None)
    
    # Define nutrition attributes based on thresholds
    nutrition_attrs = []
    
    # Key nutrients to track
    nutrient_bins = {
        'added sugar': [(0, 5, 'low'), (5, 15, 'medium'), (15, 999, 'high')],
        'calories': [(0, 100, 'low'), (100, 200, 'medium'), (200, 999, 'high')],
        'protein': [(0, 5, 'low'), (5, 15, 'medium'), (15, 999, 'high')],
        'sodium': [(0, 200, 'low'), (200, 500, 'medium'), (500, 9999, 'high')],
        'total fat': [(0, 5, 'low'), (5, 15, 'medium'), (15, 999, 'high')],
        'cholesterol': [(0, 10, 'low'), (10, 50, 'medium'), (50, 999, 'high')],
        'fiber': [(0, 3, 'low'), (3, 10, 'medium'), (10, 999, 'high')],
    }
    
    for nutrient, bins in nutrient_bins.items():
        for min_val, max_val, level in bins:
            attr_id = f"nutrition_{nutrient.replace(' ', '_')}_{level}"
            display_name = f"{nutrient.title()} - {level.title()}"
            nutrition_attrs.append({
                'attr_id': attr_id,
                'attr_family': 'nutrition',
                'display_name': display_name
            })
    
    return pd.DataFrame(nutrition_attrs)


#     attr_id                          attr_family    display_name
# brand_chobani                    brand          Chobani
# health_kosher                    health_claim   Kosher
# health_pescetarian               health_claim   Pescetarian
# health_vegan                     health_claim   Vegan
# health_vegetarian                health_claim   Vegetarian
# nutrition_added_sugar_high       nutrition      Added Sugar - High
# nutrition_calories_medium        nutrition      Calories - Medium
# nutrition_protein_high           nutrition      Protein - High