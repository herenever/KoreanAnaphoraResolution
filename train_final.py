from ELECTRADataModule import *
from ElectraAnaphoraResolution import *
import transformers 
import logging

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    model = ElectraForResolution(learning_rate=5e-6)
    dm = ResolutionDataModule(batch_size=32,train_path="./anaphora_dataset/train_V1.csv",valid_path="./anaphora_dataset/validation_V1.csv",max_length=256,doc1_col='document1',doc2_col='document2',label_col='label',ante_col='antecedent',num_workers=32)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor = 'total_Accuracy_Val',
        dirpath = './model_checkpoint',
        filename = 'version_final/{epoch:02d}--{total_Accuracy_Val:.4f}',
        verbose = True,
        save_last = True,
        mode = 'max',
        save_top_k = -1
    )
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join('./model_checkpoint','train_log_v1'),log_graph=True,default_hp_metric=False)  
    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        default_root_dir='./model_checkpoint',
        logger = tb_logger,
        callbacks = [checkpoint_callback,lr_logger],
        max_epochs = 100,
        gpus = 4
    )

    trainer.fit(model=model,datamodule=dm)
