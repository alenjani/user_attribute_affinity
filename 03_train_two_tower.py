# ================================================
# 3. TWO-TOWER HOUSEHOLD ↔ ATTRIBUTE MODEL
# ================================================

# Build lookup tables
hh_id_to_idx = {hh: i for i, hh in enumerate(hist_pd['household_id'].unique())}
n_households = len(hh_id_to_idx)
n_attrs = len(attr_id_to_idx)
embed_dim = bottleneck_dim

# Household tower: aggregate embeddings from history
class HouseholdTower(nn.Module):
    def __init__(self, n_attrs, embed_dim):
        super().__init__()
        self.attr_embed = nn.Embedding.from_pretrained(attr_embeddings, freeze=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, batch_hist_attr_idx, batch_hist_score):
        # Look up attribute embeddings
        e = self.attr_embed(batch_hist_attr_idx)   # (B, hist_len, embed_dim)
        w = batch_hist_score.unsqueeze(-1)         # (B, hist_len, 1)
        agg = (e * w).mean(dim=1)                  # weighted average
        return self.proj(agg)

# Attribute tower
class AttributeTower(nn.Module):
    def __init__(self, attr_embeddings):
        super().__init__()
        self.attr_embed = nn.Embedding.from_pretrained(attr_embeddings, freeze=False)
    def forward(self, attr_idx):
        return self.attr_embed(attr_idx)

# Full two-tower
class TwoTower(nn.Module):
    def __init__(self, household_tower, attribute_tower):
        super().__init__()
        self.hh_tower = household_tower
        self.attr_tower = attribute_tower
    def forward(self, hh_attrs, hh_scores, candidate_attrs):
        z_h = self.hh_tower(hh_attrs, hh_scores)          # (B, d)
        z_a = self.attr_tower(candidate_attrs)            # (B, d)
        return torch.sum(z_h * z_a, dim=1)                # dot product

# Instantiate towers
hh_tower = HouseholdTower(n_attrs, embed_dim)
attr_tower = AttributeTower(attr_embeddings)
model = TwoTower(hh_tower, attr_tower)

# Build synthetic training samples
def make_batch(labels_pd, hist_pd, batch_size=64, hist_len=5):
    batch = labels_pd.sample(batch_size)
    hh_attrs, hh_scores, cand_attrs, y = [], [], [], []
    for _, row in batch.iterrows():
        h, a, label = row['household_id'], row['attr_id'], row['y_binary']
        h_hist = hist_pd[hist_pd['household_id']==h].sample(hist_len, replace=True)
        hh_attrs.append([attr_id_to_idx[x] for x in h_hist['attr_id']])
        hh_scores.append(list(h_hist['hist_score']))
        cand_attrs.append(attr_id_to_idx[a])
        y.append(label)
    return (torch.tensor(hh_attrs), torch.tensor(hh_scores, dtype=torch.float32),
            torch.tensor(cand_attrs), torch.tensor(y, dtype=torch.float32))

# Training loop
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):
    hh_attrs, hh_scores, cand_attrs, y = make_batch(labels_pd, hist_pd, 128)
    optimizer.zero_grad()
    logits = model(hh_attrs, hh_scores, cand_attrs)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    print(f"Two-Tower Epoch {epoch}, Loss {loss.item():.4f}")
