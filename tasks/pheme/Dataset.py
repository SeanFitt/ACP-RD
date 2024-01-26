import imp
import torch

# task_to_keys = {
#     "pheme": ("source_content", 're_content')
# }
task_to_keys = {
    "pheme": ("sent1", 'sent2')
}


from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
import datasets
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class PhemeDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        ramdon_seed=training_args.seed
        self.is_regression=False
        #training_arg后续看看怎么传入
        task_name='charliehebdo'
        label2id={'rumor':1,'non-rumor':0}
        id2label={1:'rumor',0:'non-rumor'}
        train_data = pd.read_json('tasks/pheme/dataset/processed_data/train.json')
        eval_data = pd.read_json('tasks/pheme/dataset/processed_data/eval.json')
        test_data = pd.read_json('tasks/pheme/dataset/processed_data/test.json')

        # train_data.loc[train_data.re_content.isnull(),'re_content'] = ' '
        # test_data.loc[test_data.re_content.isnull(),'re_content'] = ' '
        # train_data = train_data.dropna()
        # debug_null=train_data.sent2.isnull()
        # for i in range(len(debug_null)):
        #     if(debug_null[i] or type(train_data.loc[i,'sent2'])!=str):
        #         print(str(i)+'is null')     
        # debug_null=eval_data.sent2.isnull()
        # for i in range(len(debug_null)):
        #     if(debug_null[i] or type(eval_data.loc[i,'sent2'])!=str):
        #         print(str(i)+'is null') 
        # debug_null=test_data.sent2.isnull()
        # for i in range(len(debug_null)):
        #     if(debug_null[i] or type(test_data.loc[i,'sent2'])!=str):
        #         print(str(i)+'is null')    
                
        # debug_null=train_data.sent1.isnull()
        # for i in range(len(debug_null)):
        #     if(debug_null[i] or type(train_data.loc[i,'sent1'])!=str):
        #         print(str(i)+'is null')     
        # debug_null=eval_data.sent1.isnull()
        # for i in range(len(debug_null)):
        #     if(debug_null[i] or type(eval_data.loc[i,'sent1'])!=str):
        #         print(str(i)+'is null') 
        # debug_null=test_data.sent1.isnull()
        # for i in range(len(debug_null)):
        #     if(debug_null[i] or type(test_data.loc[i,'sent1'])!=str):
        #         print(str(i)+'is null')   
        # train_data.loc[train_data.sent2.isnull(),'sent2'] = ' '
        # eval_data.loc[eval_data.sent2.isnull(),'sent2'] = ' '
        # test_data.loc[test_data.sent2.isnull(),'sent2'] = ' '
        # train_data.loc[train_data.sent1.isnull(),'sent1'] = ' '
        # eval_data.loc[eval_data.sent1.isnull(),'sent1'] = ' '
        # test_data.loc[test_data.sent1.isnull(),'sent1'] = ' '

        # print(train_data.sent2.isnull())

        for i in range(len(train_data)):
            train_data.loc[i,'label']=label2id[train_data.loc[i,'label']]
        for i in range(len(eval_data)):
             eval_data.loc[i,'label']=label2id[eval_data.loc[i,'label']]
        for i in range(len(test_data)):
            test_data.loc[i,'label']=label2id[test_data.loc[i,'label']]

        # train_data=train_data.dropna()
        # eval_data=eval_data.dropna()
        # test_data=test_data.dropna()

        # train_data.to_csv('tasks/pheme/dataset/processed_data/train_data.csv',mode='w',index=False,line_terminator="\r\n")
        # eval_data.to_csv('tasks/pheme/dataset/processed_data/eval_data.csv',mode='w',index=False,line_terminator="\r\n")
        # test_data.to_csv('tasks/pheme/dataset/processed_data/test_data.csv',mode='w',index=False,line_terminator="\r\n")
        # raw_datasets=load_dataset('csv',data_files={"train":"tasks/pheme/dataset/processed_data/train_data.csv",
        # "test":"tasks/pheme/dataset/processed_data/test_data.csv",
        # "validation":"tasks/pheme/dataset/processed_data/eval_data.csv"
        # })

        train_dataset = datasets.Dataset.from_pandas(train_data)
        eval_dataset = datasets.Dataset.from_pandas(eval_data)
        test_dataset = datasets.Dataset.from_pandas(test_data)

        raw_datasets=datasets.DatasetDict({'train':train_dataset,
        'validation':eval_dataset,
        'test':test_dataset})

        print(type(raw_datasets))
        #raw_datasets['train'] = HFDataset.from_pandas( train_data)
        #aw_datasets['test'] = HFDataset.from_pandas( test_data)

        self.tokenizer = tokenizer
        self.data_args = data_args
        #Glue的非回归任务的num_labels设成了1，那咱也不知道了
        self.num_labels = 2

        #你好，这里是要改的，sentence1和2从原文和回复文中取出来
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]

        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            #load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        self.raw_datasets=raw_datasets

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))


        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        #self.metric = load_metric("accuracy")
        #self.metric = load_metric("metrics/accuracy.py")
        self.metric = load_metric("f1")

        #这两句可能会出问题，第一句好像是glue专用的，第二句大概是跟编码精度有关
        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


        #可能出现的问题：
    def preprocess_function(self, examples):
        # Tokenize the texts
                args = (
                    (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
                )
                result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

                return result

        #这里改成我要的F1分数，acc，迁移得分等
    def compute_metrics(self, p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
            if self.data_args.dataset_name is not None:
                result = self.metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif self.is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {
                    "accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
                    "f1":(2*(((preds==p.label_ids)&(p.label_ids==1)/((preds==p.label_ids)&(p.label_ids==1)+(preds!=p.label_ids)&(p.label_ids==0)))*((preds==p.label_ids)&(p.label_ids==1)/((preds==p.label_ids)&(p.label_ids==1)+(preds==p.label_ids)&(p.label_ids==0)))/(((preds==p.label_ids)&(p.label_ids==1)/((preds==p.label_ids)&(p.label_ids==1)+(preds!=p.label_ids)&(p.label_ids==0)))+((preds==p.label_ids)&(p.label_ids==1)/((preds==p.label_ids)&(p.label_ids==1)+(preds==p.label_ids)&(p.label_ids==0)))))).astype(np.float32).mean().item()
                }


        