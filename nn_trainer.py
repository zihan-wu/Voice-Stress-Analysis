import torch.nn as nn
import torch
import numpy as np
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tqdm.auto import tqdm
from typing import Any, List, Dict
import pytorch_lightning as pl
from sklearn.metrics import recall_score, accuracy_score
from pathlib import Path
from pytorch_lightning.plugins import DDPPlugin

_RANDOM_SEED = 42
def serialize_value(v):
    if isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
        return v
    else:
        return str(v)

def hparams_to_json(hparams):
    return {k: serialize_value(v) for k, v in hparams.items()}

class AttentionPooling(nn.Module):
    def __init__(self, d_input, d_out):
        super().__init__()

        self.fc = nn.Linear(d_input, 1)
        self.out = nn.Linear(d_input, d_out)

    def forward(self, x, n_wins):
        att = self.fc(x)
        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :].to(att.device) < n_wins[:, None].to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")
        att = torch.nn.functional.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)
        return self.out(x)


class TransformerCLF(nn.Module):
    def __init__(self, nfeatures: int, nlabels: int, conf: Dict):
        super().__init__()

        d_model = conf["hidden_dim"]
        dim_feedforward = conf["hidden_dim"]
        num_heads = 1
        num_layers = conf["hidden_layers"]
        self.inputnorm = torch.nn.BatchNorm1d(nfeatures)

        self.linear_layer = nn.Linear(nfeatures, d_model)
        conf["initialization"](
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain('linear'),
        )
        self.relu = nn.ReLU()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))


        self.projection = AttentionPooling(d_model, nlabels)

        self.logit_loss: torch.nn.Module
        self.activation: torch.nn.Module = torch.nn.LogSoftmax(dim=1)


    def forward_logit(self, x: torch.Tensor, n_wins) -> torch.Tensor:
        x = self.inputnorm(x.transpose(1,2)).transpose(1,2)
        x = self.relu(self.linear_layer(x))
        #x = x.transpose(1,0)

        if n_wins is not None:
            mask = ~((torch.arange(x.shape[1])[None, :]).to(x.device) < n_wins[:, None].to(torch.long).to(x.device))
        else:
            mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.projection(x, n_wins)
        return x

    def forward(self, x: torch.Tensor, n_wins) -> torch.Tensor:
        x = self.forward_logit(x, n_wins)
        x = self.activation(x)
        return x


class MLPCLF(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, conf: Dict):
        super().__init__()

        hidden_modules: List[torch.nn.Module] = [torch.nn.BatchNorm1d(in_features)]
        curdim = in_features
        
        last_activation = "linear"
        for i in range(conf["hidden_layers"]):
            linear = torch.nn.Linear(curdim, conf["hidden_dim"])
            conf["initialization"](
                linear.weight,
                gain=torch.nn.init.calculate_gain(last_activation),
            )
            hidden_modules.append(linear)
            hidden_modules.append(nn.BatchNorm1d(conf["hidden_dim"]))
            hidden_modules.append(torch.nn.Dropout(0.1))
            hidden_modules.append(torch.nn.ReLU())
            curdim = conf["hidden_dim"]
            last_activation = "relu"

        self.hidden = torch.nn.Sequential(*hidden_modules)

        self.projection = torch.nn.Linear(curdim, out_features)
        self.activation = torch.nn.LogSoftmax(dim=1)


        conf["initialization"](
            self.projection.weight, gain=torch.nn.init.calculate_gain(last_activation)
        )

    def forward_logit(self, x: torch.Tensor, n_wins) -> torch.Tensor:
        x = self.hidden(x)
        x = self.projection(x)
        return x
    
    def forward(self, x: torch.Tensor, n_wins) -> torch.Tensor:
        x = self.forward_logit(x, n_wins)
        x = self.activation(x)
        return x

class VSAClassifier(pl.LightningModule):
    def __init__(
        self,
        nfeatures: int,
        nlabels: int,
        conf: Dict,
    ):
        super().__init__()

        self.save_hyperparameters()

        if conf['clf'] == 'transf':
            self.predictor = TransformerCLF(
                nfeatures, nlabels, conf
            )
        elif conf['clf'] == 'mlp':
            self.predictor = MLPCLF(
                nfeatures, nlabels, conf
            )
        else:
            raise ValueError('Invalid clf in config file')
            
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = conf['lr']
        self.weight_decay = conf['weight_decay']

    def forward(self, x, wins):
        x = self.predictor(x, wins)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        y_true, y_pred= self._step(batch, batch_idx)
        targets = torch.argmax(y_true, dim=1)
        assert targets.requires_grad == False, 'target has grad'
        loss = self.loss(y_pred, y_true) # must be pred then true
        score= accuracy_score(targets.detach().cpu().numpy(), 
        torch.argmax(y_pred, dim=1).detach().cpu().numpy())
        # Logging to TensorBoard by default
        self.log("train_score", score, prog_bar=True, logger=True)
        return loss

    def _step(self, batch, batch_idx):
        x, y_true, wins = batch
        y_pred = self.predictor.forward_logit(x, wins)
        return y_true.type(torch.float), y_pred

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    # def training_epoch_end(self, outputs):
    #     scores = [out['score'] for out in outputs]
    #     self.log("train_score", np.mean(scores), prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs: List):

        outputs = list(zip(*outputs))

        if outputs[0][0].dim() > 1:
            y_true = torch.vstack(outputs[0])
            y_pred = torch.vstack(outputs[1])
        else:
            y_true = torch.hstack(outputs[0])
            y_pred = torch.hstack(outputs[1])

        targets = torch.argmax(y_true, dim=1)
        loss = self.loss(y_pred.detach(), y_true.detach())
        score = accuracy_score(targets.detach().cpu().numpy(), 
            torch.argmax(y_pred, dim=1).detach().cpu().numpy())
            
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_score", score, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y_true, wins = batch
        y_pred = self.forward(x, wins)
        return torch.argmax(y_true, dim=1), torch.argmax(y_pred, dim=1)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer



def train_model(clf, conf, train_dl, validation_dl, test_dl):
    checkpoint_callback = ModelCheckpoint(monitor='val_score', 
                                          mode='max',
                                          every_n_epochs = 1)
    early_stop_callback = EarlyStopping(
        monitor='val_score',
        min_delta=0.00,
        patience=conf["patience"],
        check_on_train_epoch_end=False,
        verbose=False,
        mode='max',
    )

    logger = CSVLogger(Path("logs"))
    logger.log_hyperparams(hparams_to_json(conf))

    if conf['gpus'] > 0:
        accelerator = 'gpu' 
        devices = conf['gpus']
    else:
        accelerator = 'cpu'
        devices = None

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=accelerator,
        devices=devices,
        check_val_every_n_epoch=conf["check_val_every_n_epoch"],
        max_epochs=conf["max_epochs"],
        min_epochs=conf["min_epochs"],
        deterministic=True,
        strategy=DDPPlugin(find_unused_parameters=False),
        logger=logger,
    )

    trainer.fit(clf, train_dl, validation_dl)

    print('BEST MODEL PATH IS {}, with score {}'.format(checkpoint_callback.best_model_path, checkpoint_callback.best_model_score))
    if checkpoint_callback.best_model_score is not None:
        return {'trainer': trainer, 
                'model_path': checkpoint_callback.best_model_path, 
                'val_score':checkpoint_callback.best_model_score.detach().cpu().item(), 
                'conf': conf}
    else:
        raise ValueError(
            f"No score {checkpoint_callback.best_model_score} for this model"
        )
    



def hyparam_search(nfeatures, nlabels, confs, train_dl, validation_dl, test_dl):

    results = []
    for confi, conf in tqdm(enumerate(confs), desc="grid"):
        print(f"Training configuration {confi+1} of {len(confs)}: {conf}")
        clf = VSAClassifier(nfeatures, nlabels, conf)
        result = train_model(clf, conf, train_dl, validation_dl, test_dl)
        results.append(result)
        print(f"Validation Score of configuration {confi+1}: {result['val_score']}")
    
    sorted_result = sorted(results, key=lambda d: d['val_score'])
    
    return sorted_result[-1]

def test_pred(result, test_dl):

    #conf = result['conf']
    epoch = torch.load(result['model_path'])["epoch"]
    trainer = result['trainer']
    trainer.fit_loop.current_epoch = epoch

    test_results = trainer.predict(
        ckpt_path=result['model_path'], dataloaders=test_dl
    )

    y_true, y_pred = zip(*test_results)
    y_true = torch.hstack(y_true)
    y_pred = torch.hstack(y_pred)
    return y_true, y_pred