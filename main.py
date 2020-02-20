if __name__ == "__main__":
    from pytorch_lightning import loggers
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    from trainer import LanguageModelTrainer
    from utils import DotDict
    from os import path

    if path.exists('.comet.config'):
        import configparser
        comet_config = configparser.ConfigParser()

        comet_config.read('.comet.config')

        logger = loggers.CometLogger(
            api_key=comet_config['comet']['api_key'],
            project_name="transformer-chatbot-test",
            workspace="luungoc2005"
        )

    else:
        logger = loggers.TensorBoardLogger()

    checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/')

    config = DotDict({
        'model_type': 'transformer',
        'ninp': 256, 
        'nhead': 4, 
        'nhid': 1024,
        'nlayers': 12, 
        'tie_layers': True,
        'tie_encoder_decoder': True,
        'dropout': 0.1,
        'lr': 3e-4,
        'num_warmup_steps': 10000,
        'batch_size': 32,
        'accumulate_grad_batches': 2,
        'bptt': 384
    })
    model = LanguageModelTrainer(config)

    trainer = Trainer(
        gradient_clip_val=.5,
        gpus=1,
        use_amp=True,
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        logger=logger,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model)