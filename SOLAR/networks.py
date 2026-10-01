import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Convenzione: tutti i modelli prendono input [batch, window, 1]
# e (tranne dove indicato) restituiscono [batch, window, 1], come in Keras.
# Internamente le conv lavorano in [batch, canali, window].


# ----------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------
class Conv1dSame(nn.Module):
    """Conv1d con padding 'same' identico a Keras/TF (kernel pari incluso:
    il padding in più va a destra). Il padding è fatto con F.pad esplicito,
    così nell'ONNX non compare auto_pad (che esp-ppq non gestisce)."""

    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size - 1 - self.pad_left
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad_left, self.pad_right)))


# ----------------------------------------------------------------------------
# CNN (seq2seq, "dense" finali implementati come Conv1d k=1 -> niente permute
# in mezzo alla rete, più comodo per l'export su ESP32)
# ----------------------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self, kernels=(10, 6, 5), channels=(30, 40, 50), hidden=128):
        super().__init__()
        layers, in_ch = [], 1
        for k, c in zip(kernels, channels):
            layers += [Conv1dSame(in_ch, c, k), nn.ReLU()]
            in_ch = c
        self.conv_layers = nn.Sequential(*layers)
        # Dense(128, relu) + Dense(1) applicati per timestep
        self.dense1 = nn.Linear(in_ch, hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # [B, 1, W]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = self.dense1(x)
        x = self.relu(x)
        return self.dense2(x)


# ----------------------------------------------------------------------------
# CRNN
# ----------------------------------------------------------------------------
class CRNNBlock(nn.Module):
    def __init__(self, in_ch, filters, kernel, drop_out):
        super().__init__()
        self.conv = Conv1dSame(in_ch, filters, kernel)
        # eps/momentum come Keras (torch momentum = 1 - keras momentum)
        self.bn = nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01)
        self.act = nn.ReLU()
        # MaxPooling1D(pool_size=1) in Keras è un no-op: omesso
        self.drop = nn.Dropout(drop_out)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class CRNNModel(nn.Module):
    def __init__(self, drop_out=0.1, kernel=5, num_layers=3, gru_units=64):
        super(CRNNModel, self).__init__()
        blocks, in_ch = [], 1
        for i in range(num_layers):
            filters = 2 ** (i + 5)
            blocks.append(CRNNBlock(in_ch, filters, kernel, drop_out))
            in_ch = filters
        self.blocks = nn.Sequential(*blocks)
        self.bi_direct = nn.GRU(filters, gru_units, bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(gru_units * 2, 512)
        self.relu = nn.ReLU()
        self.frame_level = nn.Linear(512, 1)


    def forward(self, x):
        x = x.permute(0, 2, 1)          # [B, 1, W]
        x = self.blocks(x)
        x = x.permute(0, 2, 1)          # [B, W, C]
        x, _ = self.bi_direct(x)
        x = self.dense1(x)
        x = self.relu(x)
        return self.frame_level(x)               # [B, W, 1]


# ----------------------------------------------------------------------------
# LightCNN
# ----------------------------------------------------------------------------
class LightCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1dSame(1, 20, 8)
        # Keras PReLU: alpha inizializzato a 0 (qui uno per canale)
        self.prelu1 = nn.PReLU(20, init=0.0)
        self.conv2 = Conv1dSame(20, 20, 6)
        self.prelu2 = nn.PReLU(20, init=0.0)
        self.dense1 = nn.Conv1d(20, 144, 1)   # Dense(144, linear)
        self.dense2 = nn.Conv1d(144, 1, 1)    # Dense(1, linear)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.dense2(self.dense1(x))
        return x.permute(0, 2, 1)

