import os
import shutil
import pickle as pk
import datetime
from typing_extensions import Self
from tqdm import tqdm

import pandas as pd

import torch
import torch.optim as optim

from data import Generator, build_datasets, BatchDataLoader
from utils import load_config, save_config
from models import factory as model_factory
from train_test import Regularizer, test, comp_contrast, sparse_hess, train_epoch

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
dev = torch.device(dev)

torch.manual_seed(0)

def main(cfg):
    res = []
    now = datetime.datetime.now()
    res_dir = f'res/{now:%Y%m%d%H%M}_{cfg["name"]}'
    if os.path.exists(res_dir): shutil.rmtree(res_dir)
    os.mkdir(res_dir)

    print('Save config...')
    save_config(cfg, f'{res_dir}/cfg.yml')

    for gen_cfg in tqdm(cfg['generators'], position=0):
        k = gen_cfg['k']
        l = gen_cfg['l']
        m = gen_cfg['m']

        generator = Generator(**gen_cfg)

        for data_cfg in tqdm(cfg['datasets'], position=1, leave=False):
            # print('Build data sets...')
            n = data_cfg['n']
            # bs = sampling['bs']

            datasets = build_datasets(generator, **data_cfg)

            for model_cfg in tqdm(cfg['models'], position=2, leave=False):
                # print('Build model...')
                model = model_factory(model_cfg['model'], k, l, m, **model_cfg.get('kwargs', {})).to(dev)
                
                checkpoint = model_cfg.get('load', None)
                if checkpoint is not None:
                    model.load_state_dict(torch.load(checkpoint))
                
                # print('Build dataloader, regularizer, optimizer...')
                train_cfg = model_cfg['train']

                train_ldr, test_ldr_id, test_ldr_ood, test_ldr_rand = \
                    [BatchDataLoader(dataset, train_cfg.get('batch_size', 64)) for dataset in datasets]
                
                regularizers = [Regularizer(**reg_cfg, l=l) for reg_cfg in train_cfg['regularizers']] if 'regularizers' in model_cfg else None
                
                lr = train_cfg['lr']
                lr_steps = train_cfg['lr_steps']
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)                

                log_dict = {'n data': n, 'k': k, 'l': l, 'm': m, 'sampling': data_cfg['display_name'], 'model': model_cfg['display_name']}

                # print('Train model...')
                for epoch in tqdm(range(train_cfg['epochs']), position=3, leave=False):
                    if epoch in lr_steps:
                        lr /= 3
                        for g in optimizer.param_groups:
                            g['lr'] = lr

                    # train
                    loss = train_epoch(model, train_ldr, optimizer, regularizers=regularizers)

                    # test
                    jac_id = test(model, test_ldr_id, comp_contrast, l=l)
                    jac_ood = test(model, test_ldr_ood, comp_contrast, l=l)
                    # hes2_id = test(mdl, id_ldr, sparse_hess, p=2)
                    # hes2_ood = test(mdl, ood_ldr, sparse_hess, p=2)
                    r2_id = test(model, test_ldr_id)
                    r2_ood = test(model, test_ldr_ood)
                    r2_both = test(model, test_ldr_rand)

                    # log results
                    log_dict.update({'n samples': (epoch+1)*n})
                    _res = dict(log_dict)
                    _res.update({'metric': 'train loss', 'domain': 'ID', 'val': loss})
                    res.append(_res)

                    _res = dict(log_dict)
                    _res.update({'metric': 'compositional contrast', 'domain': 'ID', 'val': jac_id})
                    res.append(_res)
                    _res = dict(log_dict)
                    _res.update({'metric': 'compositional contrast', 'domain': 'OOD', 'val': jac_ood})
                    res.append(_res)

                    _res = dict(log_dict)
                    _res.update({'metric': 'test R²', 'domain': 'ID', 'val': r2_id})
                    res.append(_res)
                    _res = dict(log_dict)
                    _res.update({'metric': 'test R²', 'domain': 'OOD', 'val': r2_ood})
                    res.append(_res)
                    _res = dict(log_dict)
                    _res.update({'metric': 'test R²', 'domain': 'BOTH', 'val': r2_both})
                    res.append(_res)
                    # res.append({'n data': n, 'n samples': (epoch+1)*n, 'k': setting['k'], 'l': setting['l'], 'm': setting['m'], 'sampling': sampling['name'], 'model': model['name'], 'metric': 'sparse Hessian L2', 'domain': 'ID', 'val': hes2_id})
                    # res.append({'n data': n, 'n samples': (epoch+1)*ns, 'k': setting['k'], 'l': setting['l'], 'm': setting['m'], 'sampling': sampling['name'], 'model': model['name'], 'metric': 'sparse Hessian L2', 'domain': 'OOD', 'val': hes2_ood})
                
                save_path = f"{res_dir}/{gen_cfg['name']}_{data_cfg['name']}_{model_cfg['name']}.pth"
                print(f'Save model {save_path} ...')
                torch.save(model.state_dict(), save_path)

                print('Save results...')
                res_df = pd.DataFrame.from_dict(res)
                with open(f'{res_dir}/df.pkl', 'wb') as f:
                    pk.dump(res_df, f)
    
    return res_df


if __name__=='__main__':
    cfg = load_config('cfg.yaml')
    
    main(cfg)