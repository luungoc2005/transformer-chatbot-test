import os
import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from utils import DotDict, get_batch

from models import TransformerModel, LSTMModel
from data import RedditCorpus

SAVE_PATH = './models/'

def save_model(model, hparams):
    from os import path
    import json

    model_path = path.join(SAVE_PATH, 'model_state.pt')
    config_path = path.join(SAVE_PATH, 'model_params.json')

    torch.save(model.state_dict(), model_path)
    
    with open(config_path, 'w') as config_file:
        json.dump(hparams, config_file)


class LMCorpusDataset(Dataset):

    def __init__(self, split_name='train', batch_size=64, bptt=128):
        super(LMCorpusDataset, self).__init__()

        self.corpus = RedditCorpus(split_name=split_name) \
            .batchify(batch_size)
        self.bptt = bptt

    def __len__(self):
        return self.corpus.size(0) // self.bptt

    def __getitem__(self, index):
        return get_batch(self.corpus, index, self.bptt)

class LanguageModelTrainer(pl.LightningModule):

    def __init__(self, hparams=DotDict({
        'model_type': 'transformer',
        'ninp': 128, 
        'nhead': 2, 
        'nhid': 512, 
        'nlayers': 2, 
        'tie_layers': True,
        'tie_encoder_decoder': True,
        'dropout': 0.1,
    })):
        super(LanguageModelTrainer, self).__init__()

        self.hparams = hparams if isinstance(hparams, DotDict) \
                        else DotDict(hparams)

        from utils import get_default_tokenizer
        self.vocab_size = get_default_tokenizer()._tokenizer.get_vocab_size()

        self.model_type = hparams.get('model_type', 'transformer')
        assert self.model_type in ['transformer', 'lstm']

        if self.model_type == 'transformer':
            self.model = TransformerModel(ntoken=self.vocab_size, **hparams)
        else:
            self.model = LSTMModel(ntoken=self.vocab_size, **hparams)
        
        self.batch_size = hparams.get('batch_size', 64)
        self.bptt = hparams.get('bptt', 128)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch[0][0], batch[1][0]

        output = self.forward(data)

        loss = F.cross_entropy(
            output.view(-1, self.vocab_size), targets
        )

        tensorboard_logs = {
            'train_loss': loss,
            'ppl': torch.exp(loss)
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, targets = batch[0][0], batch[1][0]

        output = self.forward(data)

        loss = F.cross_entropy(
            output.view(-1, self.vocab_size), targets
        )
        ppl = torch.exp(loss)

        tensorboard_logs = {
            'val_loss': loss,
            'val_ppl': ppl
        }

        return {'val_loss': loss, 'val_ppl': ppl, 'log': tensorboard_logs}


    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_ppl = torch.stack([x['val_ppl'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_ppl': avg_ppl}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        from radam import RAdam

        optimizer = RAdam(self.parameters(), 
            lr=self.hparams.get('lr', 3e-4),
            betas=(0.9, 0.980)
        )

        self.scheduler = None

        return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()

        num_warmup_steps=self.hparams.get('num_warmup_steps', 10000)
        accumulate_grad_batches=self.hparams.get('accumulate_grad_batches', 1)

        if batch_nb % accumulate_grad_batches == 0 and num_warmup_steps > 0:
            if self.scheduler is None:
                def lr_lambda(current_step):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1.0, num_warmup_steps))
                    else:
                        # return 1
                        # decay
                        return num_warmup_steps ** .5 * current_step ** -0.5

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

            self.scheduler.step()

        if batch_nb % 5000 == 0: # save every 5000 steps
            save_model(self.model, self.hparams)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            LMCorpusDataset(
                split_name='train',
                batch_size=self.batch_size,
                bptt=self.bptt
            ),
            batch_size=1,
            shuffle=True
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            LMCorpusDataset(
                split_name='val',
                batch_size=self.batch_size,
                bptt=self.bptt
            ),
            batch_size=1,
            shuffle=True
        )