import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from config import EMB_DIM, NUM_EMB, NUM_ARTISTS, LR


class Net(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.context_emb = nn.Embedding(num_embeddings=NUM_EMB, embedding_dim=emb_dim)
        self.track_emb = nn.Embedding(num_embeddings=NUM_EMB, embedding_dim=emb_dim)

        self.artist_context_emb = nn.Embedding(num_embeddings=NUM_ARTISTS, embedding_dim=emb_dim)
        self.artist_track_emb = nn.Embedding(num_embeddings=NUM_ARTISTS, embedding_dim=emb_dim)

    def forward(self, x):
        context = self.context_emb(x[:, 0])
        track = self.track_emb(x[:, 1])
        artist_context = self.artist_context_emb(x[:, 2])
        artist_track = self.artist_track_emb(x[:, 3])

        context += artist_context
        track += artist_track

        return torch.sum(context * track, dim=1)


class ContextualRanker(pl.LightningModule):
    def __init__(self, embedding_dim=10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = Net(embedding_dim)
        self.save_hyperparameters()

    def forward(self, x):
        return self.net.forward(x)

    def step(self, batch, batch_idx, metric, prog_bar=False):
        x, y = batch
        predictions = self.forward(x)
        loss = F.mse_loss(predictions, y.float(), reduction='mean')
        self.log(metric, loss, prog_bar=prog_bar)
        return loss

    def test_step(self, batch, batch_idx, prog_bar=False):
        x, y = batch
        predictions = self.forward(x)
        targets = y[:, 0].float()
        avgs = y[:, 1].float()
        rdms = y[:, 2].float()

        loss = F.mse_loss(predictions, targets, reduction='mean')
        avg_loss = F.mse_loss(avgs, targets, reduction='mean')
        rdm_loss = F.mse_loss(rdms, targets, reduction='mean')

        self.log("test_loss", loss, prog_bar=prog_bar)
        self.log("avg_loss", avg_loss, prog_bar=prog_bar)
        self.log("rdm_loss", rdm_loss, prog_bar=prog_bar)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val_loss", True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

