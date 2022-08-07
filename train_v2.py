from ELECTRADataModule import *
from ElectraAnaphoraResolution_v2 import *
import transformers 
import logging

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    # logging.getLogger("transformers.tokenization_utils_base").disabled = True
    # logging.getLogger("transformers.tokenization_utils").disabled = True
    model = ElectraForResolution_v2(learning_rate=5e-6)
    dm = ResolutionDataModule(batch_size=32,train_path="./anaphora_dataset/train.csv",valid_path="./anaphora_dataset/validation.csv",max_length=128,doc1_col='document1',doc2_col='document2',label_col='label',ante_col='antecedent',num_workers=32)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor = 'total_Accuracy_Val',
        dirpath = './model_checkpoint',
        filename = 'version_4/{epoch:02d}--{total_Accuracy_Val:.4f}',
        verbose = True,
        save_last = True,
        mode = 'max',
        save_top_k = -1
    )
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join('./model_checkpoint','tb_logs_v4'),log_graph=True,default_hp_metric=False)  
    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        default_root_dir='./model_checkpoint',
        logger = tb_logger,
        callbacks = [checkpoint_callback,lr_logger],
        max_epochs = 50,
        gpus = 4
    )

    trainer.fit(model=model,datamodule=dm)
