# ================================================
# 2. AUTOENCODER FOR ATTRIBUTE EMBEDDINGS
# ================================================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, bottleneck_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# Train AE
input_dim = feature_dim
bottleneck_dim = 16
ae = AutoEncoder(input_dim, 64, bottleneck_dim)
optimizer = optim.Adam(ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

X = torch.tensor(attr_feat_matrix, dtype=torch.float32)
for epoch in range(50):
    optimizer.zero_grad()
    recon, z = ae(X)
    loss = criterion(recon, X)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"AE Epoch {epoch}, Loss {loss.item():.4f}")

# Final attribute embeddings (normalized)
with torch.no_grad():
    _, Z = ae(X)
    attr_embeddings = torch.nn.functional.normalize(Z, p=2, dim=1)
