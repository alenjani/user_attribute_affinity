# build_attr_vocab.py
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import RegexTokenizer, Word2Vec, VectorAssembler, StandardScaler
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("BuildAttrVocab").getOrCreate()

# ================
# 0) Load inputs
# ================
attr = spark.read.parquet("/data/dim_attribute_vocab_base")   # attr_id, attr_family, display_name, threshold_rule_json
mapia = spark.read.parquet("/data/map_item_attribute")        # upc_id, attr_id, p_attr, evidence_sources
items = spark.read.parquet("/data/dim_item")                  # upc_id, department_nm, section_nm, brand_name, ...
nutri = spark.read.parquet("/data/fact_item_nutrition")       # upc_id, calories, protein_g, carbs_g, fiber_g, sugar_g, added_sugar_g, fat_g, sodium_mg

# ----------------
# Helper: safe division
# ----------------
def safe_div(col_num, col_den):
    return F.when(col_den.isNull() | (col_den == 0), F.lit(0.0)).otherwise(col_num / col_den)

# =======================================================
# 1) DIRECT DESCRIPTORS (family flags + rule meta + text)
# =======================================================

# 1a) Family one-hots
families = ["brand","dietary","nutrition"]
for fam in families:
    attr = attr.withColumn(f"fam_{fam}", (F.col("attr_family") == F.lit(fam)).cast("double"))

# 1b) Simple rule meta features (length, operators)
#    If you store rule JSONs, we add crude features for now (presence/length)
attr = attr.withColumn("has_rule", F.col("threshold_rule_json").isNotNull().cast("double")) \
           .withColumn("rule_len", F.when(F.col("threshold_rule_json").isNull(), F.lit(0))
                                 .otherwise(F.length("threshold_rule_json")).cast("double"))

# 1c) Text embedding of attribute display_name (cheap but effective)
tok = RegexTokenizer(inputCol="display_name", outputCol="tokens", pattern="\\W+", toLowercase=True)
attr_tok = tok.transform(attr.fillna({"display_name": ""}))

w2v = Word2Vec(vectorSize=32, minCount=1, inputCol="tokens", outputCol="name_vec")
w2v_model = w2v.fit(attr_tok)
attr_desc = w2v_model.transform(attr_tok).drop("tokens")

# Direct feature vector (raw, we’ll scale later)
direct_cols = ["fam_brand","fam_dietary","fam_nutrition","has_rule","rule_len","name_vec"]

# =======================================================
# 2) AGGREGATED ITEM ENRICHMENT (nutrition / diversity)
# =======================================================

# 2a) Join map → nutrition to compute per-attr nutrient stats (weighted by p_attr)
mia_n = mapia.join(nutri, "upc_id", "left")

# Weighted sums
agg_nutri = (mia_n
    .groupBy("attr_id")
    .agg(
        F.sum("p_attr").alias("w_sum"),
        F.sum(F.col("p_attr") * F.col("calories")).alias("w_calories"),
        F.sum(F.col("p_attr") * F.col("protein_g")).alias("w_protein"),
        F.sum(F.col("p_attr") * F.col("carbs_g")).alias("w_carbs"),
        F.sum(F.col("p_attr") * F.col("fiber_g")).alias("w_fiber"),
        F.sum(F.col("p_attr") * F.col("sugar_g")).alias("w_sugar"),
        F.sum(F.col("p_attr") * F.col("added_sugar_g")).alias("w_added_sugar"),
        F.sum(F.col("p_attr") * F.col("fat_g")).alias("w_fat"),
        F.sum(F.col("p_attr") * F.col("sodium_mg")).alias("w_sodium"),
        F.countDistinct("upc_id").alias("num_items_with_attr")
    ))

# Turn into weighted means (guarded)
nutri_means = (agg_nutri
    .withColumn("avg_calories",    safe_div(F.col("w_calories"), F.col("w_sum")))
    .withColumn("avg_protein_g",   safe_div(F.col("w_protein"), F.col("w_sum")))
    .withColumn("avg_carbs_g",     safe_div(F.col("w_carbs"), F.col("w_sum")))
    .withColumn("avg_fiber_g",     safe_div(F.col("w_fiber"), F.col("w_sum")))
    .withColumn("avg_sugar_g",     safe_div(F.col("w_sugar"), F.col("w_sum")))
    .withColumn("avg_added_sugar", safe_div(F.col("w_added_sugar"), F.col("w_sum")))
    .withColumn("avg_fat_g",       safe_div(F.col("w_fat"), F.col("w_sum")))
    .withColumn("avg_sodium_mg",   safe_div(F.col("w_sodium"), F.col("w_sum")))
    .select("attr_id","num_items_with_attr",
            "avg_calories","avg_protein_g","avg_carbs_g","avg_fiber_g",
            "avg_sugar_g","avg_added_sugar","avg_fat_g","avg_sodium_mg")
)

# 2b) Diversity / popularity (join to item taxonomy)
mia_i = mapia.join(items.select("upc_id","department_nm","section_nm"), "upc_id", "left")

div_pop = (mia_i
    .groupBy("attr_id")
    .agg(
        F.countDistinct("upc_id").alias("item_coverage"),
        F.countDistinct("department_nm").alias("dept_diversity"),
        F.countDistinct("section_nm").alias("section_diversity")
    ))

# 2c) Combine aggregated features
attr_agg = (nutri_means.join(div_pop, "attr_id", "full_outer")
    .fillna({
        "num_items_with_attr":0,
        "item_coverage":0,
        "dept_diversity":0,
        "section_diversity":0,
        "avg_calories":0.0,"avg_protein_g":0.0,"avg_carbs_g":0.0,"avg_fiber_g":0.0,
        "avg_sugar_g":0.0,"avg_added_sugar":0.0,"avg_fat_g":0.0,"avg_sodium_mg":0.0
    })
)

# =======================================================
# 3) MERGE DIRECT + AGG FEATURES → feature_vec
# =======================================================

# Join descriptor & aggregates
wide = attr_desc.select("attr_id","attr_family","display_name","name_vec","fam_brand","fam_dietary","fam_nutrition","has_rule","rule_len") \
       .join(attr_agg, "attr_id", "left")

# Assemble numeric columns
num_cols = [
    "fam_brand","fam_dietary","fam_nutrition","has_rule","rule_len",
    "num_items_with_attr","item_coverage","dept_diversity","section_diversity",
    "avg_calories","avg_protein_g","avg_carbs_g","avg_fiber_g",
    "avg_sugar_g","avg_added_sugar","avg_fat_g","avg_sodium_mg"
]
assembler_direct = VectorAssembler(inputCols=num_cols, outputCol="direct_num_vec")
wide = assembler_direct.transform(wide)

# Concatenate name_vec (dense) + direct_num_vec
# Trick: VectorAssembler can take vector columns too
assembler_all = VectorAssembler(inputCols=["name_vec","direct_num_vec"], outputCol="raw_feature_vec")
wide = assembler_all.transform(wide)

# Optional: scale numeric stability (keeps name_vec scale too)
scaler = StandardScaler(inputCol="raw_feature_vec", outputCol="feature_vec", withMean=True, withStd=True)
scaler_model = scaler.fit(wide)
attr_vocab = scaler_model.transform(wide).select("attr_id","feature_vec")

# =======================================================
# 4) SAVE
# =======================================================
attr_vocab.write.mode("overwrite").parquet("/data/attr_vocab.parquet")
