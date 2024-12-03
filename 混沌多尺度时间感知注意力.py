import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from scipy.stats import entropy

class StandardScaler:
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        return self

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-7)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return (data * (self.std + 1e-7)) + self.mean

class DataLoaderCustom:
    def __init__(self, file_path, target_col, seq_len=24, pred_len=12, test_size=0.3):
        self.file_path = file_path
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.test_size = test_size
        self.scaler = StandardScaler()

    def load_data(self):
        df = pd.read_excel(self.file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        data = df.values

        data = self.scaler.fit_transform(data)

        x, y, time_diffs = self.create_sequences(data)

        x_train, x_test, y_train, y_test, time_train, time_test = train_test_split(x, y, time_diffs, test_size=self.test_size, random_state=128)

        return x_train, x_test, y_train, y_test, time_train, time_test

    def create_sequences(self, data):
        x, y, time_diffs = [], [], []
        for i in range(len(data) - self.seq_len - self.pred_len):
            seq_x = data[i:i + self.seq_len]
            time_diff = np.arange(self.seq_len, 0, -1)
            lyap = self.calculate_lyapunov(seq_x[:, 0])
            fractal = self.calculate_fractal_dimension(seq_x[:, 0])
            kolmogorov = self.calculate_kolmogorov_entropy(seq_x[:, 0])
            seq_x = np.hstack((seq_x, np.full((self.seq_len, 1), lyap), np.full((self.seq_len, 1), fractal), np.full((self.seq_len, 1), kolmogorov)))
            x.append(seq_x)
            y.append(data[i + self.seq_len:i + self.seq_len + self.pred_len, 0])
            time_diffs.append(time_diff)
        return np.array(x), np.array(y).reshape(-1, self.pred_len), np.array(time_diffs)

    def calculate_lyapunov(self, series, embedding_dim=2, tau=1, max_neighbors=20):
        N = len(series)
        X = np.array([series[i: i + embedding_dim * tau: tau] for i in range(N - embedding_dim * tau)])
        M = len(X)
        distances = np.array([np.linalg.norm(X - X[i], axis=1) for i in range(M)])
        neighbors = np.sort(distances, axis=1)[:, 1:max_neighbors+1]
        with np.errstate(divide='ignore', invalid='ignore'):
            divergence = np.mean(np.log(neighbors / neighbors[:, 0][:, None]), axis=1)
        lyapunov_exponent = np.mean(divergence[np.isfinite(divergence)])
        return lyapunov_exponent

    def calculate_fractal_dimension(self, series, k=2):
        N = len(series)
        X = np.array([series[i: i + k] for i in range(N - k)])
        dists = pdist(X)
        counts, bin_edges = np.histogram(dists, bins=100)
        cumsum_counts = np.cumsum(counts)
        log_r = np.log(bin_edges[1:])
        log_C = np.log(cumsum_counts)
        slope, _, _, _, _ = linregress(log_r, log_C)
        fractal_dimension = slope
        return fractal_dimension

    def calculate_kolmogorov_entropy(self, series):
        N = len(series)
        X = np.array([series[i:i + 2] for i in range(N - 2)])
        dists = pdist(X)
        counts, bin_edges = np.histogram(dists, bins=100)
        probabilities = counts / np.sum(counts)
        kolmogorov_entropy = entropy(probabilities)
        return kolmogorov_entropy

class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, embed_type, freq, dropout):
        super(DataEmbedding_inverted, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.value_embedding = None
        self.position_embedding = nn.Embedding(freq, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        B, L, D = x.size()
        if self.value_embedding is None:
            self.value_embedding = nn.Linear(D, self.d_model)
        x = self.value_embedding(x)
        if x_mark is not None:
            x = x + self.position_embedding(x_mark)
        return self.dropout(x)

class TimeAwareFullAttention(nn.Module):
    def __init__(self, mask_flag, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(TimeAwareFullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor  # 控制时间衰减的因子
        self.scale = scale or 1.0 / (factor ** 0.5)
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, time_diffs, attn_mask=None):
        B, H, L, E = queries.shape
        _, _, S, _ = keys.shape
        scale = self.scale or 1.0 / np.sqrt(E)

        # Applying time decay based on time differences
        decay = torch.exp(-time_diffs / self.factor).unsqueeze(1).unsqueeze(-1)  # Adjust shape to broadcast
        scores = torch.einsum("bhle,bhse->bhls", queries, keys) * decay

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bhse->bhle", A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class TimeAwareAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(TimeAwareAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, time_diffs, attn_mask=None):
        if len(queries.shape) == 3:
            queries = queries.unsqueeze(1)
            keys = keys.unsqueeze(1)
            values = values.unsqueeze(1)

        B, H, L, E = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, H, L, E // H)
        keys = self.key_projection(keys).view(B, H, S, E // H)
        values = self.value_projection(values).view(B, H, S, E // H)

        out, attn = self.inner_attention(queries, keys, values, time_diffs, attn_mask)
        out = out.view(B, L, E)

        return self.out_projection(out), attn

class TimeAwareEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(TimeAwareEncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, time_diffs, attn_mask=None):
        new_x, attn = self.attention(x, x, x, time_diffs, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        x = x + self.dropout(self.activation(self.ffn(x)))
        x = self.norm2(x)
        return x, attn

class TimeAwareEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(TimeAwareEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, time_diffs, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, time_diffs, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class Configs:
    def __init__(self, pred_len):
        self.seq_len = 24
        self.pred_len = pred_len
        self.d_model = 64
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 128
        self.dropout = 0.1
        self.activation = "gelu"
        self.task_name = "long_term_forecast"
        self.embed = "timeF"
        self.freq = 24
        self.output_attention = False
        self.num_class = 1
        self.enc_in = 1
        self.factor = 5

class ChaosTransformer(nn.Module):
    def __init__(self, configs):
        super(ChaosTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = TimeAwareEncoder(
            [
                TimeAwareEncoderLayer(
                    TimeAwareAttentionLayer(
                        TimeAwareFullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forward(self, x_enc, time_diffs, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, time_diffs, mask)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.projection(enc_out)
            dec_out = dec_out[:, -self.pred_len:, 0]  # 确保输出形状为 [batch_size, pred_len]
            return (dec_out, attns) if self.output_attention else dec_out
        elif self.task_name == 'imputation':
            dec_out = self.projection(enc_out)
            return (dec_out, attns) if self.output_attention else dec_out
        elif self.task_name == 'anomaly_detection':
            dec_out = self.projection(enc_out)
            return (dec_out, attns) if self.output_attention else dec_out
        elif self.task_name == 'classification':
            enc_out = enc_out.view(enc_out.size(0), -1)
            dec_out = self.dropout(self.act(self.projection(enc_out)))
            return (dec_out, attns) if self.output_attention else dec_out
        return None

def plot_results(true_values, predicted_values, title):
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Values')
    plt.plot(predicted_values, label='Predicted Values')
    plt.title(title)
    plt.legend()
    plt.show()

def train_and_predict(pred_len, title):
    data_loader = DataLoaderCustom(r"C:\Users\7iCK\Desktop\10_features.xlsx", target_col="Price", seq_len=30, pred_len=pred_len)
    x_train, x_test, y_train, y_test, time_train, time_test = data_loader.load_data()

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(time_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), torch.tensor(time_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    configs = Configs(pred_len)
    model = ChaosTransformer(configs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch, t_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            t_batch = t_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch, t_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    model.eval()
    test_loss = 0.0
    true_values = []
    predicted_values = []
    with torch.no_grad():
        for x_batch, y_batch, t_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            t_batch = t_batch.to(device)
            outputs = model(x_batch, t_batch)
            test_loss += loss_fn(outputs, y_batch).item()
            true_values.extend(y_batch.cpu().numpy())
            predicted_values.extend(outputs.cpu().numpy())

    true_values = np.array(true_values).reshape(-1)
    predicted_values = np.array(predicted_values).reshape(-1)

    print(f"Test Loss: {test_loss / len(test_loader)}")
    print(f"Test MSE: {mean_squared_error(true_values, predicted_values)}")
    print(f"Test R²: {r2_score(true_values, predicted_values)}")

    plot_results(true_values, predicted_values, title=title)

if __name__ == "__main__":
    train_and_predict(pred_len=1, title="Next Day Prediction")
    train_and_predict(pred_len=7, title="Next Week Prediction")
    train_and_predict(pred_len=30, title="Next Month Prediction")
