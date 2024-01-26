import numpy as np

class result_helper():
    def __init__(data_set,possibility,round):
        out_data=data_set.to_pandas()
        pos_0=possibility[:][0]
        pos_1=possibility[:][1]
        out_data.loc[:]['pos_0']=pos_0
        out_data.loc[:]['pos_1']=pos_1
        ##################
        # 基于margin的不确定分
        margin=np.abs(np.diff(possibility,axis=1)).flatten()
        ##################
        out_data.loc[:]['margin']=margin
        out_data.to_excel('tasks/pheme/result/confident_r'+str(round)+'.xls')