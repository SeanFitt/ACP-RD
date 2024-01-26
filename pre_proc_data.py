import pymysql
import pandas as pd
import sklearn
import random
import pandas as pd

from sqlalchemy import create_engine

def pre_procsee_data():
    # MYSQL_HOST = 'sh-cynosdbmysql-grp-7do2iolu.sql.tencentcdb.com'
    # MYSQL_PORT = '28621'
    # MYSQL_USER = 'zhejie'
    # MYSQL_PASSWORD = 'szj19970905,.,.'
    # MYSQL_DB = 'ForExam'

    # engine = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8mb4'
    #                         % (MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB))

    raw_data=pd.read_json('tasks/pheme/dataset/processed_data/train_labeled.json')
    raw_data=raw_data.fillna('')
    # raw_data.to_sql(name='tmpTable',
    #                 con=engine,
    #                 if_exists='replace')
    raw_data['sent']=raw_data['sent1']+raw_data['sent2']

    # db = pymysql.connect(
    #     host="sh-cynosdbmysql-grp-7do2iolu.sql.tencentcdb.com",
    #     port=28621,
    #     user="zhejie", 
    #     passwd="szj19970905,.,.", 
    #     database="ForExam",
    #     charset='utf8' )

    # raw_data=pd.read_sql(
    #     'select * from tmpTable',
    #     db
    # )
    raw_data=raw_data.drop(columns=['id'])
    # raw_data.rename(columns={'domain':'task_name','post':'sent'},inplace=True)

    task_list=raw_data['task_name'].unique()
    label_list=['rumor','non-rumor']
    pairs=pd.DataFrame(columns=['sent1','sent2','hard_neg','task_name','label'])
    
    #同领域同标签下生成正样本
    for i in task_list:
        for j in label_list:
            #index的问题
            pos_1=raw_data.query('task_name=='+str(i)+'& label=="'+str(j)+'"').reset_index(drop=True)
            pos_1=pos_1.sample(frac=30.0,replace=True).reset_index(drop=True)
            pos_2=pos_1.sample(frac=1).reset_index(drop=True)
            for k in range(len(pos_1)):
                pairs.loc[len(pairs)]=[pos_1.loc[k]['sent'],
                                    pos_2.loc[k]['sent'],
                                    None,i,j]
    #同标签为正样本
    # for i in label_list:
    #     #index的问题
    #     pos_1=raw_data.query('label == @i').reset_index(drop=True)
    #     pos_1=pos_1.sample(frac=50.0,replace=True).reset_index(drop=True)
    #     pos_2=pos_1.sample(frac=1.0).reset_index(drop=True)
    #     for k in range(len(pos_1)):
    #         pairs.loc[len(pairs)]=[pos_1.loc[k,'sent'],
    #                             pos_2.loc[k,'sent'],
    #                             None,None,i]            
    
        
    print('Unmodified pairs,',pairs.shape)
    pairs=pairs.drop(pairs[pairs['sent1']==pairs['sent2']].index)
    pairs=pairs.reset_index(drop=True)
    print('Modified pairs,',str(pairs.shape))         

    #非同标签且非同领域构成负样本
    # for i in range(len(pairs)):
    #     task_name=pairs.loc[i,'task_name']
    #     label=pairs.loc[i,'label']
    #     neg_list=raw_data.query('label != @label and task_name != @task_name').reset_index(drop=True)
    #     random_neg_index=random.randint(0,len(neg_list)-1)
    #     pairs.loc[i,'hard_neg']=neg_list.loc[random_neg_index,'sent']
    
    #非同标签或非同领域构成副样本
    for i in range(len(pairs)):
        task_name=pairs.loc[i]['task_name']
        label=pairs.loc[i]['label']
        neg_list=raw_data.query('task_name!="'+str(task_name)+'"or label!="'+label+'"').reset_index(drop=True)
        random_neg_index=random.randint(0,len(neg_list)-1)
        pairs.loc[i,'hard_neg']=neg_list.loc[random_neg_index,'sent']
    
    #非同标签构成副样本
    # for i in range(len(pairs)):
    #     label=pairs.loc[i]['label']
    #     neg_list=raw_data.query('label!="'+label+'"').reset_index(drop=True)
    #     random_neg_index=random.randint(0,len(neg_list)-1)
    #     pairs.loc[i]['hard_neg']=neg_list.loc[random_neg_index]['sent']
        
    pairs=pairs.drop(columns=['task_name','label'])
    pairs.to_csv('tasks/pheme/dataset/processed_data/for_sp_simcse.csv',index=False)

    new_pairs=pd.read_csv('tasks/pheme/dataset/processed_data/for_sp_simcse.csv')
    return(new_pairs.shape)

if __name__=='__main__':
    a=pre_procsee_data()