import pandas as pd

# ================================================
# 4. GENERATE HH ↔ ATTRIBUTE AFFINITY SCORES
# ================================================

def score_households(model, hist_pd, hh_id_to_idx, attr_id_to_idx, top_k=5, hist_len=5):
    results = []

    for hh in hh_id_to_idx.keys():
        # Sample history for this HH
        h_hist = hist_pd[hist_pd['household_id']==hh]
        if h_hist.empty:
            continue
        # pad/trim to hist_len
        h_hist = h_hist.sample(hist_len, replace=True, random_state=42)
        hh_attrs = torch.tensor([[attr_id_to_idx[x] for x in h_hist['attr_id']]])
        hh_scores = torch.tensor([list(h_hist['hist_score'])], dtype=torch.float32)

        # Get household embedding
        z_h = model.hh_tower(hh_attrs, hh_scores)  # (1, d)

        # Get all attribute embeddings
        all_attr_idx = torch.arange(len(attr_id_to_idx))
        z_a = model.attr_tower(all_attr_idx)       # (A, d)

        # Dot products → affinity scores
        scores = torch.matmul(z_a, z_h.T).squeeze().detach().numpy()
        
        # Normalize with sigmoid to get 0–1 affinity
        probs = 1 / (1 + np.exp(-scores))

        # Take top-K
        top_idx = np.argsort(probs)[-top_k:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            attr_name = list(attr_id_to_idx.keys())[list(attr_id_to_idx.values()).index(idx)]
            results.append((hh, attr_name, float(probs[idx]), rank))

    return pd.DataFrame(results, columns=["household_id","attr_id","affinity_score","rank"])


# Run scoring
df_affinity = score_households(model, hist_pd, hh_id_to_idx, attr_id_to_idx, top_k=5)

print(df_affinity.head(20))
