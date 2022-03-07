import torch
from torch.utils.data import DataLoader

import logging
from datetime import datetime
from math import sqrt

from cormorant.models import CormorantQM9
from cormorant.models.autotest import cormorant_tests

from cormorant.engine import Engine
from cormorant.engine import init_argparse, init_file_paths, init_logger, init_cuda
from cormorant.engine import init_optimizer, init_scheduler
from cormorant.data.utils import initialize_datasets

from cormorant.data.collate import collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')

def main():

    # Initialize arguments -- Just
    args = init_argparse('qm9')

    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloader
    args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'qm9', subtract_thermo=args.subtract_thermo,
                                                                    force_download=args.force_download
                                                                    )

    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_fn)
                         for split, dataset in datasets.items()}
    
    for batch_idx, data in enumerate(dataloaders['train']):
        print(data.keys())
        break
    
    print('Done')

if __name__ == '__main__':
    main()

