# ================================================
# 1. PYSPARK DATA GENERATION (synthetic pipeline)
# ================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, when, col, explode, lit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Create Spark session
spark = SparkSession.builder.appName("HHAttrAffinity").getOrCreate()

# Parameters
N_households = 5000
N_attrs = 50    # e.g., mix of brand + dietary + nutrition bins
N_hist = 10000  # number of household-attribute interactions
feature_dim = 20

# --- Generate synthetic attribute table with random feature vectors ---
attr_ids = [f"attr_{i}" for i in range(N_attrs)]
attr_features = np.random.rand(N_attrs, feature_dim).astype(np.float32)

df_attrs = spark.createDataFrame(
    [(attr_ids[i], attr_features[i].tolist()) for i in range(N_attrs)],
    ["attr_id", "feature_vec"]
)

# --- Generate synthetic household history table ---
hh_ids = [f"hh_{i}" for i in range(N_households)]
hist_data = []
rng = np.random.default_rng(123)
for _ in range(N_hist):
    h = rng.choice(hh_ids)
    a = rng.choice(attr_ids)
    score = rng.uniform(0.1, 1.0)
    hist_data.append((h, a, float(score)))

df_hist = spark.createDataFrame(hist_data, ["household_id", "attr_id", "hist_score"])

# --- Generate synthetic labels for next-window prediction ---
# Binary outcome if household "buys" attr next time
label_data = []
for _ in range(N_hist):
    h = rng.choice(hh_ids)
    a = rng.choice(attr_ids)
    y = rng.integers(0, 2)
    label_data.append((h, a, int(y)))

df_labels = spark.createDataFrame(label_data, ["household_id", "attr_id", "y_binary"])

# Collect to pandas/numpy for PyTorch
attrs = df_attrs.collect()
attr_id_to_idx = {row['attr_id']: i for i, row in enumerate(attrs)}
attr_feat_matrix = np.stack([row['feature_vec'] for row in attrs])

labels_pd = df_labels.toPandas()
hist_pd = df_hist.toPandas()
