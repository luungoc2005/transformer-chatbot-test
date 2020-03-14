
from utils import DotDict
# config = DotDict({
#     'model_type': 'transformer',
#     'ninp': 256, 
#     'nhead': 4, 
#     'nhid': 1024,
#     'nlayers': 6,
#     'tie_layers': True,
#     'tie_encoder_decoder': False,
#     'dropout': 0.1,
#     'lr': 3e-4,
#     'num_warmup_steps': 16000,
#     'batch_size': 32,
#     'accumulate_grad_batches': 12,
#     'bptt': 512
# })

config = DotDict({
    'model_type': 'transformer',
    'emb_size': 256,
    'ninp': 256,
    'nhid': 1024,
    'nlayers': 6,
    'nhead': 4,
    'dropout': 0.1,
    'tie_encoder_decoder': False,
    'tie_layers': True,
    'lr': 0.0056,
    'min_lr': 0,
    'adam_eps': 1e-9,
    'num_warmup_steps': 3125,
    'weight_decay': .01,
    'batch_size': 64,
    'accumulate_grad_batches': 2,
    'bptt': 128
})

if __name__ == "__main__":
    from pytorch_lightning import loggers
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    from trainer import LanguageModelTrainer
    from os import path

    import random

    if path.exists('.comet.config'):
        import configparser
        comet_config = configparser.ConfigParser()

        comet_config.read('.comet.config')

        logger = loggers.CometLogger(
            api_key=comet_config['comet']['api_key'],
            project_name="lstm-nmt-chatbot-test",
            workspace="luungoc2005"
        )

        for k, v in config.items():
            logger.experiment.log_parameter(k, v)
    else:
        logger = loggers.TensorBoardLogger()

    checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/')

    model = LanguageModelTrainer(config)

    # MODEL_PATH = './models/model_state.pt'
    # if path.exists(MODEL_PATH):
    #     import torch
    #     print('Loading from checkpoint')
    #     model.model.load_state_dict(torch.load(MODEL_PATH))

    trainer = Trainer(
        gradient_clip_val=0.5,
        gpus=1,
        use_amp=True,
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        logger=logger,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model)