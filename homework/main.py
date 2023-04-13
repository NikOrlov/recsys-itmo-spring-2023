import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, BatchSizeFinder
import config
import model
import dataset

if __name__ == '__main__':

    model_name = '2emb'
    pl_model = model.ContextualRanker(config.EMB_DIM)

    train_data = pd.read_csv(os.path.join(config.DATA_PATH, 'train_data_artist.csv'))
    val_data = pd.read_csv(os.path.join(config.DATA_PATH, 'val_data_artist.csv'))
    test_data = pd.read_csv(os.path.join(config.DATA_PATH, 'test_data_artist.csv'))
    features = ['start', 'track', 'artist_context', 'artist_track']

    dm = dataset.ContextualRankerData(train_data=train_data,
                                      val_data=val_data,
                                      test_data=test_data,
                                      features=features)

    checkpoint_callback = ModelCheckpoint(dirpath=config.CHECKPOINT_PATH,
                                          monitor='val_loss',
                                          mode='min',
                                          save_last=True,
                                          save_top_k=1,
                                          filename=model_name + config.CHECKPOINT_FILENAME)
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.STOPPING_PATIENCE)
    callbacks = [checkpoint_callback,
                 early_stopping,
                 ]

    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS,
                         accelerator=config.ACCELERATOR,
                         devices=config.DEVICES,
                         callbacks=callbacks,
                         default_root_dir=config.LOG_PATH)

    trainer.fit(pl_model, datamodule=dm)
    pl.Trainer(devices=1).test(pl_model, datamodule=dm)

    best = model.ContextualRanker.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(best, datamodule=dm)
