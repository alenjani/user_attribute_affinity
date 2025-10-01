# etl/build_map_item_attribute.py (UPDATED)

def build_map_item_attribute(item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create long-form item→attribute mapping.
    """
    import ast
    
    mappings = []
    
    # 1. Brand mapping (same as before)
    brand_map = item_df[['upc_nbr', 'iri_brands_nm']].copy()
    brand_map = brand_map[brand_map['iri_brands_nm'].notna()]
    brand_map['attr_id'] = 'brand_' + brand_map['iri_brands_nm'].str.replace(' ', '_').str.lower()
    brand_map['p_attr'] = 1.0
    brand_map['evidence_sources'] = 'iri_brands_nm'
    mappings.append(brand_map[['upc_nbr', 'attr_id', 'p_attr', 'evidence_sources']])
    
    # 2. Category mapping (same as before)
    cat_map = item_df[['upc_nbr', 'smic_category_cd']].copy()
    cat_map = cat_map[cat_map['smic_category_cd'].notna()]
    cat_map['attr_id'] = 'category_' + cat_map['smic_category_cd'].astype(str)
    cat_map['p_attr'] = 1.0
    cat_map['evidence_sources'] = 'smic_category_cd'
    mappings.append(cat_map[['upc_nbr', 'attr_id', 'p_attr', 'evidence_sources']])
    
    # 3. Class mapping (same as before)
    class_map = item_df[['upc_nbr', 'smic_class_cd']].copy()
    class_map = class_map[class_map['smic_class_cd'].notna()]
    class_map['attr_id'] = 'class_' + class_map['smic_class_cd'].astype(str)
    class_map['p_attr'] = 1.0
    class_map['evidence_sources'] = 'smic_class_cd'
    mappings.append(class_map[['upc_nbr', 'attr_id', 'p_attr', 'evidence_sources']])
    
    # 4. Department mapping (same as before)
    dept_map = item_df[['upc_nbr', 'department_nm']].copy()
    dept_map = dept_map[dept_map['department_nm'].notna()]
    dept_map['attr_id'] = 'dept_' + dept_map['department_nm'].str.replace(' ', '_').str.lower()
    dept_map['p_attr'] = 1.0
    dept_map['evidence_sources'] = 'department_nm'
    mappings.append(dept_map[['upc_nbr', 'attr_id', 'p_attr', 'evidence_sources']])
    
    # 5. Health claims (UPDATED - handle multi-value)
    health_map = item_df[['upc_nbr', 'health_clain_type_dsc']].copy()
    health_map = health_map[health_map['health_clain_type_dsc'].notna()]
    
    # Split and explode
    health_map['health_clain_type_dsc'] = health_map['health_clain_type_dsc'].str.split('/')
    health_map = health_map.explode('health_clain_type_dsc')
    health_map['health_clain_type_dsc'] = health_map['health_clain_type_dsc'].str.strip()
    
    health_map['attr_id'] = 'health_' + health_map['health_clain_type_dsc'].str.replace(' ', '_').str.lower()
    health_map['p_attr'] = 1.0
    health_map['evidence_sources'] = 'health_clain_type_dsc'
    mappings.append(health_map[['upc_nbr', 'attr_id', 'p_attr', 'evidence_sources']])
    
    # 6. Shelf mapping (same as before)
    shelf_map = item_df[['upc_nbr', 'shelf_nm']].copy()
    shelf_map = shelf_map[shelf_map['shelf_nm'].notna()]
    shelf_map['attr_id'] = 'shelf_' + shelf_map['shelf_nm'].str.replace(' ', '_').str.lower()
    shelf_map['p_attr'] = 1.0
    shelf_map['evidence_sources'] = 'shelf_nm'
    mappings.append(shelf_map[['upc_nbr', 'attr_id', 'p_attr', 'evidence_sources']])
    
    # 7. Nutrition attributes (NEW)
    nutrition_map = build_nutrition_mapping(item_df)
    mappings.append(nutrition_map[['upc_nbr', 'attr_id', 'p_attr', 'evidence_sources']])
    
    # Combine
    map_item_attribute = pd.concat(mappings, ignore_index=True)
    
    # Deduplicate
    map_item_attribute = map_item_attribute.drop_duplicates(subset=['upc_nbr', 'attr_id'])
    
    return map_item_attribute


def build_nutrition_mapping(item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map items to nutrition attribute buckets.
    
    Example: Item with 22g added sugar → "nutrition_added_sugar_high"
    """
    import ast
    
    # Parse nutrition
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
    
    # Define binning rules
    nutrient_bins = {
        'added sugar': [(0, 5, 'low'), (5, 15, 'medium'), (15, 999, 'high')],
        'calories': [(0, 100, 'low'), (100, 200, 'medium'), (200, 999, 'high')],
        'protein': [(0, 5, 'low'), (5, 15, 'medium'), (15, 999, 'high')],
        'sodium': [(0, 200, 'low'), (200, 500, 'medium'), (500, 9999, 'high')],
        'total fat': [(0, 5, 'low'), (5, 15, 'medium'), (15, 999, 'high')],
        'cholesterol': [(0, 10, 'low'), (10, 50, 'medium'), (50, 999, 'high')],
        'fiber': [(0, 3, 'low'), (3, 10, 'medium'), (10, 999, 'high')],
    }
    
    mappings = []
    
    for idx, row in item_df.iterrows():
        upc = row['upc_nbr']
        nutrition = row['nutrition_dict']
        
        for nutrient, bins in nutrient_bins.items():
            value = nutrition.get(nutrient, None)
            
            if value is None:
                continue
            
            # Find which bin this value falls into
            for min_val, max_val, level in bins:
                if min_val <= value < max_val:
                    attr_id = f"nutrition_{nutrient.replace(' ', '_')}_{level}"
                    mappings.append({
                        'upc_nbr': upc,
                        'attr_id': attr_id,
                        'p_attr': 1.0,
                        'evidence_sources': 'nutrition_nm'
                    })
                    break
    
    return pd.DataFrame(mappings)



# upc_nbr          attr_id                      p_attr  evidence_sources
# 0001234567890    brand_chobani                1.0     iri_brands_nm
# 0001234567890    health_kosher                1.0     health_clain_type_dsc
# 0001234567890    health_vegan                 1.0     health_clain_type_dsc
# 0001234567890    health_vegetarian            1.0     health_clain_type_dsc
# 0001234567890    nutrition_added_sugar_high   1.0     nutrition_nm
# 0001234567890    nutrition_calories_low       1.0     nutrition_nm
# 0001234567890    nutrition_protein_medium     1.0     nutrition_nm