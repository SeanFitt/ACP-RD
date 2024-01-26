import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from tasks.pheme.Dataset import PhemeDataset

#获得带有Prefix或者prompt结构的模型
from model.utils import get_model, TaskType
#from tasks.glue.dataset import GlueDataset

#看起来像是训练时候的指标评估
from training.trainer_base import BaseTrainer

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset=PhemeDataset(tokenizer, data_args, training_args)

    config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )

    #拿到了Prefix或者Prompt形式的model
    model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        test_key='f1',
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
    )

    return trainer, dataset.raw_datasets["test"]
