import json
from math import fabs
import torch
from tasks.pheme.MyEncoder import MyEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score



task_to_keys = {
    "pheme": ("sent1", 'sent2')
}

#from torch.utils import data
#from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
#from datasets import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding,default_data_collator
from transformers.trainer_utils import EvalPrediction

import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)



class al_PhemeDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args,round) -> None:
        super().__init__()
        ramdon_seed=training_args.seed
        label2id={'rumor':1,'non-rumor':0}
        id2label={1:'rumor',0:'non-rumor'}
        self.is_regression=False
        #training_arg后续看看怎么传入
        #task_name='all_data'
        #test_set_name='all_data'
        #init_set
        data_path='tasks/pheme/dataset/processed_data/'

        init_set=data_args.init_set
        
        #如果模型经过0次训练，采取随机采样的形式获得数据
        if round==0:
            raw_dataset=pd.read_json(data_path+'init.json')
            for i in range(len(raw_dataset)):
                raw_dataset.loc[i,'label']=label2id[raw_dataset.loc[i,'label']]
            train_data=raw_dataset
            eval_data=raw_dataset
            test_data=raw_dataset
            map_data=raw_dataset
            # labeled_data=pd.DataFrame(columns=['id','sent1','sent2','label','time','task_name'])
            # maxsize=len(raw_dataset)
            # index=np.random.randint(0,maxsize-1,init_set)
            # for i in range(len(index)):
            #     # print(index[i])
            #     #print(raw_dataset.loc[index[i]])
            #     #labeled data有点问题
            #     labeled_data.loc[len(labeled_data)]=raw_dataset.loc[index[i]]
            # unlabeled_data=raw_dataset.drop(raw_dataset.loc[index].index)
            # unlabeled_data.reset_index(drop=True).to_json(data_path+'train_unlabeled.json')
            # labeled_data.reset_index(drop=True).to_json(data_path+'train_labeled.json')
            
            # unlabeled_data.to_json(data_path+'train_unlabeled.json')
            # labeled_data.to_json(data_path+'train_labeled.json')
            # unlabeled_data=json.dumps(labeled_data,cls=MyEncoder, ensure_ascii=False)
            # unlabeled_data=json.dumps(unlabeled_data,cls=MyEncoder, ensure_ascii=False)
            # with open(data_path+'train_unlabeled.json','w') as f:
            #     f.write(unlabeled_data)
            # with open(data_path+'train_labeled.json','w') as f:
            #     f.write(labeled_data)
            #后面几轮写入labeled_data时，注意将head置为False      
        else:
            map_data=pd.read_json(data_path+'init.json')
            train_data=pd.read_json(data_path+'train_labeled.json')
            ##########################
            #暂且采取Eval和test相同
            eval_data=pd.read_json(data_path+'train_unlabeled.json')
            test_data=pd.read_json(data_path+'train_unlabeled.json')
            for i in range(len(train_data)):
                train_data.loc[i,'label']=label2id[train_data.loc[i,'label']]
            for i in range(len(eval_data)):
                eval_data.loc[i,'label']=label2id[eval_data.loc[i,'label']]
            for i in test_data.loc[:,'label'].index:
                test_data.loc[i,'label']=label2id[test_data.loc[i,'label']]
            for i in map_data.loc[:,'label'].index:
                map_data.loc[i,'label']=label2id[map_data.loc[i,'label']]
        train_dataset=Dataset.from_pandas(train_data,preserve_index=False)
        eval_dataset=Dataset.from_pandas(eval_data,preserve_index=False)
        test_dataset=Dataset.from_pandas(test_data,preserve_index=False)
        map_dataset=Dataset.from_pandas(map_data,preserve_index=False)

        raw_datasets=DatasetDict({'train':train_dataset,
        'validation':eval_dataset,
        'test':test_dataset,
        'map':map_dataset})
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
            load_from_cache_file=not data_args.overwrite_cache,
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

        #self.metric = load_metric("f1")
        self.metric = load_metric("metrics/"+data_args.metric+".py")

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
                #调试用，之后删除
                # for i in range(len(args)):
                #     a=args[i][0]
                #     b=args[i][1]
                #     if(type(a)!=str or type(b)!=str):
                #         bk=1
                result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

                return result

        #这里改成我要的F1分数，acc，迁移得分等
    # def compute_metrics(self, p: EvalPrediction):
    #         preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #         preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
    #         if self.data_args.dataset_name is not None:
    #             result = self.metric.compute(predictions=preds, references=p.label_ids)
    #             if fabs(result[self.data_args.metric]) < 1e-6:
    #                 bk=1
    #             if len(result) > 1:
    #                 result["combined_score"] = np.mean(list(result.values())).item()
    #             return result
    #         elif self.is_regression:
    #             return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    #         else:
    #             return {
    #                 "accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
    #                 }
    def compute_metrics(self, p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
            accuracy = accuracy_score(p.label_ids, preds)
            precision = precision_score(p.label_ids, preds, average='weighted')
            recall = recall_score(p.label_ids, preds, average='weighted')
            f1 = f1_score(p.label_ids, preds, average='weighted')
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }



        
    