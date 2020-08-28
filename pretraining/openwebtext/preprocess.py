import logging
import logging
import math
import multiprocessing
import os
import random
import tarfile
from dataclasses import dataclass
from itertools import chain
from functools import partial
from pathlib import Path

import numpy as np

import torch
import torch.utils.data

from pretraining.openwebtext import arg
from pretraining.openwebtext import tokenization


logger = logging.getLogger(__name__)


def parse_tokenizer(tokenizer, text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


def create_tokenizer(vocab_file, do_lower_case=True):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    return partial(parse_tokenizer, tokenizer)


def preprocess_owt(tokenizer, src_dir, tmp_dir, trg_dir, n_dataset_building_processes, n_tensors_per_file, max_seq_length=None):
    # Preamble
    logger.info(f'Writing features to {trg_dir}.')
    os.makedirs(trg_dir, exist_ok=False)

    # Crunch files
    trg_dir = Path(trg_dir)
    src_dir = Path(src_dir)
    tmp_dir = Path(tmp_dir)
    archives = os.listdir(src_dir)
    n_archives_per_job = math.ceil(len(archives) / n_dataset_building_processes)
    job_archives = [
        archives[i * n_archives_per_job : (i + 1) * n_archives_per_job]
        for i in range(n_dataset_building_processes)
    ]

    logger.info(f'Processing {len(archives)} archives.')
    assert len(archives) > 0

    if n_dataset_building_processes == 1:
        feature_set_paths = preprocess_owt_job(tokenizer, src_dir, tmp_dir, trg_dir, job_archives, n_tensors_per_file, max_seq_length, job_id=0)
    else:
        pool = multiprocessing.Pool(processes=n_dataset_building_processes)
        preprocess_owt_job_partial = partial(preprocess_owt_job, tokenizer, src_dir, tmp_dir, trg_dir, job_archives, n_tensors_per_file, max_seq_length)
        feature_sets = pool.map(preprocess_owt_job_partial, range(n_dataset_building_processes))
        feature_set_paths = [file_path for feature_set in feature_sets for file_path in feature_set]

    return feature_set_paths


def preprocess_owt_job(tokenizer, src_dir, tmp_dir, trg_dir, job_archives, n_tensors_per_file, max_seq_length, job_id=0):
    '''
    OpenWebText is saved under the following format:
    openwebtext.zip
        |-> archive_xxx.zip
            |-> file_xxx.txt
            |-> file_xxz.txt
            ...
        |-> archive_xxz.zip
            |-> file_xxy.txt
            ...
        ...
    '''

    # Preamble
    os.makedirs(tmp_dir, exist_ok=True)

    # Process
    feature_index = 0
    feature_set_paths = []
    features = []
    for archive_id, archive in enumerate(job_archives[job_id]):
        if os.path.isdir(src_dir / archive):
            logger.info(f'Ignoring rogue directory {src_dir / archive}.')
            continue

        logger.info(f'Job {job_id}: Processing {archive_id}/{len(job_archives[job_id])} {src_dir / archive}.')

        with tarfile.open(src_dir / archive) as t:
            extracted_archive = tmp_dir / f'{archive}-extracted'
            t.extractall(extracted_archive)

        for file in os.listdir(extracted_archive):
            file_path = extracted_archive / file

            with open(file_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) > 2:
                        encoding = tokenizer(line)
                        features.append(torch.tensor(encoding))

        while len(features) > n_tensors_per_file:
            feature_set_path = trg_dir / f'feature_set_{job_id}_{feature_index}.pt'
            torch.save(features[:n_tensors_per_file], feature_set_path)
            features = features[n_tensors_per_file:]
            feature_index += 1
            feature_set_paths.append(feature_set_path)

    # Serialize
    if len(features) > 0:
        feature_set_path = trg_dir / f'feature_set_{job_id}_{feature_index}.pt'
        torch.save(features, feature_set_path)
        feature_set_paths.append(feature_set_path)

    return feature_set_paths


@dataclass(frozen=True)
class Args:
    src_dir: arg.Str = 'data/openwebtext'
    trg_dir: arg.Str = 'data/openwebtext_features'
    tmp_dir: arg.Str = '/tmp/owt'
    vocab_file: arg.Str = 'data/vocab.txt'
    n_dataset_building_processes: arg.Int = 32
    n_tensors_per_file: arg.Int = 2048
    max_seq_length: arg.Int = 128


def main():
    args = arg.parse_to(Args)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    tokenizer = create_tokenizer(args.vocab_file)
    preprocess_owt(tokenizer=tokenizer, src_dir=args.src_dir, tmp_dir=args.tmp_dir, trg_dir=args.trg_dir, n_dataset_building_processes=args.n_dataset_building_processes, n_tensors_per_file=args.n_tensors_per_file, max_seq_length=args.max_seq_length)


if __name__ == '__main__':
    main()