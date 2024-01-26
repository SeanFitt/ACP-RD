import logging
import os
import sys
import numpy as np
from typing import Dict
import pandas as pd
import subprocess

import datasets
import transformers
from transformers.utils.dummy_pt_objects import Trainer
from transformers.trainer_utils import get_last_checkpoint,set_seed
from rocket_message_helper import rocket_message
from tasks.pheme.get_al_trainer import get_al_trainer

from arguments import get_args

from tasks.utils import *

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

#import xlsxwriter

def run_contrastive(path):
    os.chdir('../SimCSE')
    subprocess.call('sh '+path, shell=True)
    os.chdir('../prompt-tuning-v2')
#
def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

def result_helper(data_set,possibility,round,for_map=False):
    if not for_map:
        out_data=data_set.to_pandas()
        #后续这里因为多标签，需要根据pos进行排序
        pos_0=possibility[:,0]
        pos_1=possibility[:,1]
        out_data.loc[:,'pos_0']=pos_0
        out_data.loc[:,'pos_1']=pos_1
        ##################
        # 基于margin的不确定分
        margin=np.abs(np.diff(possibility,axis=1)).flatten()
        ##################
        out_data.loc[:,'margin']=margin
        out_data.to_excel('tasks/pheme/result/confident_r'+str(round)+'.xlsx',index=False)
    #这里是构建datamap的
    else:
        out_data=data_set.to_pandas()
        #后续这里因为多标签，需要根据pos进行排序
        pos_0=possibility[:,0]
        pos_1=possibility[:,1]
        out_data.loc[:,'pos_0']=pos_0
        out_data.loc[:,'pos_1']=pos_1
        out_data=out_data.rename(columns={'id':'guid','label':'gold'})
        out_data.to_json('tasks/pheme/result/map_r'+str(round)+'.json')
        
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #trainer.save_model()

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics()

def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):
        
        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        # 从这里下手做active learning的样本抽取工作
        # 用还没有argmax的predictions做最小置信度计算
        # 这边考虑一下归一化的影响，需不需要对其进行softmax


        predictions = np.argmax(predictions, axis=1)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
def map_data(trainer,taskname,map_data_set,per_active_budget,round):
    predictions, labels, metrics = trainer.predict(map_data_set, metric_key_prefix="predict")
    result_helper(map_data_set,predictions,round,True)
    
    

def active_data(trainer,taskname,active_data_set,per_active_budget,round):
    predictions, labels, metrics = trainer.predict(active_data_set, metric_key_prefix="predict")

    possibility=softmax(predictions,axis=1)
    ##################
    # 基于margin的不确定分
    margin=np.abs(np.diff(possibility,axis=1)).flatten()
    ##################
    #导出excel结果
    result_helper(active_data_set,possibility,round)

    index=margin.argsort()[:per_active_budget]
    data_path='tasks/pheme/dataset/processed_data/'
    unlabeled_data=pd.read_json(data_path+'train_unlabeled.json')
    # labeled_data=pd.DataFrame(columns=['id','post','label','domain'])
    labeled_data=pd.DataFrame(columns=['id','sent1','sent2','label','time','task_name'])
    for i in range(len(index)):
        labeled_data.loc[len(labeled_data)]=unlabeled_data.loc[index[i]]
    labeled_data.to_excel('tasks/pheme/result/r'+str(round)+'_choosen.xlsx',index=False)
    unlabeled_data=unlabeled_data.drop(unlabeled_data.loc[index].index)
    #unlabeled_data=json.dumps(unlabeled_data, ensure_ascii=False)
    # with open(data_path+'train_unlabeled.json','w') as f:
    #     f.write(unlabeled_data)
    unlabeled_data.reset_index(drop=True).to_json(data_path+'train_unlabeled.json')
    fname=data_path+'train_labeled.json'
    if round==0:
        labeled_data.reset_index(drop=True).to_json(data_path+'train_labeled.json')
        # with open(data_path+'train_labeled.json','w') as f:
        #     f.write(unlabeled_data)
    else:
        source_labeled_data=pd.read_json(data_path+'train_labeled.json')
        new_labeled_data=pd.concat([source_labeled_data,labeled_data])   
        new_labeled_data.reset_index(drop=True).to_json(data_path+'train_labeled.json')     
        # with open(data_path+'train_labeled.json','w') as f:
        #     f.write(unlabeled_data)
    
    
if __name__ == '__main__':
    os.system('rm -rf pre_trained_model/bert-base-uncased/*')
    os.system('rm -rf pre_trained_model/sp-simcse-bert-base-uncased/*')
    os.system('rm -rf pre_trained_model/unsp_bert_based_uncased_pheme/*')
    os.system('cp -r ../SimCSE/bert-base-uncased/* pre_trained_model/')
    global_best_score=''
    args = get_args()
    #详细参数划分见argument.py
    _, data_args, training_args, _ = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "rumor_detection":
        assert data_args.dataset_name.lower() in RUMOR_DETECTION_DATASETS
        from tasks.pheme.get_trainer import get_trainer

    else:
        raise NotImplementedError('Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)

    last_checkpoint = None
    #这里好像是定义了是否要重新训练
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )   

    #之后改成传参形式
    domain_name='all_data'

    if data_args.do_active_learning:
        print('______________pre_proc_data________________')
        os.system('python3 pre_proc_data.py')
        print('______________End pre_proc_data________________')
        print('______________run_unsup_pheme________________')
        run_contrastive('run_unsup_pheme.sh')
        print('______________End_run_unsup_pheme________________')
        
        round=int(data_args.active_budget/data_args.per_active_budget)
        #Active Learning evaluation
        trainer, predict_dataset, map_dataset=get_al_trainer(args,round=0)   
 
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        possibility=softmax(predictions,axis=1)
        margin=np.abs(np.diff(possibility,axis=1)).flatten()
        index=margin.argsort()[:data_args.init_set]
        data_path='tasks/pheme/dataset/processed_data/'
        unlabeled_data=pd.read_json(data_path+'all.json')
        init_data=pd.read_json(data_path+'init.json')

        result_helper(predict_dataset,possibility,0)
        # labeled_data=pd.DataFrame(columns=['id','post','label','domain'])
        labeled_data=pd.DataFrame(columns=['id','sent1','sent2','label','time','task_name'])

        for i in range(len(index)):
            labeled_data.loc[len(labeled_data)]=init_data.loc[index[i]]
        unlabeled_data=unlabeled_data.drop(unlabeled_data.loc[index].index)

        unlabeled_data.reset_index(drop=True).to_json(data_path+'train_unlabeled.json')
        fname=data_path+'train_labeled.json'
        labeled_data.reset_index(drop=True).to_json(data_path+'train_labeled.json')
        #obal_best_score+=('\nActive Round'+str(0)+':'+str(metrics))
        with open('tasks/pheme/result/mertic_r0.txt',mode='w') as f:
                f.write(str(metrics))
                f.close()     
                result_helper(predict_dataset,possibility,round=0)
    else:
        round=1

    for i in range(round):
        print('______________run_sup_pheme________________')
        run_contrastive('run_sup_pheme.sh')
        print('______________End_run_sup_pheme________________')
        #Active Learning需要进行多次训练  
        map_dataset=None         
        if data_args.do_active_learning:
            trainer, predict_dataset,map_dataset=get_al_trainer(args,round=i+1)
            train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
        else:
            trainer, predict_dataset=get_trainer(args)
            train(trainer, training_args.resume_from_checkpoint, last_checkpoint)

        if data_args.do_active_learning:
            #taskname之后改成传参形式
            active_data(trainer,taskname=domain_name,
            active_data_set=predict_dataset,
            per_active_budget=data_args.per_active_budget,round=i+1)
            
            map_data(trainer,domain_name,map_dataset,data_args.per_active_budget,i+1)

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        best_metric_result=str(trainer.best_metrics)
        print("__________Active Round"+str(i+1)+" End_____________ ")
        global_best_score+=('\nActive Round'+str(i+1)+':'+str(metrics))

    try:
        rocket_message().send_Message(message='ALPT train done.'+global_best_score)
    except Exception as e :
        print(e)
    finally :
        with open('result.txt','w') as f:
            f.write(global_best_score)
        
   