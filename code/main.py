import os
import pandas as pd
import warnings
from datetime import datetime
from code.utils import get_price_after
import numpy as np
warnings.filterwarnings("ignore")

class NF(object):
    def __init__(self,time_file='../data/input/time/NF_time.csv',
                 price_folder='../data/input/price/XAUUSD_NF_1min/',
                 price_prefix='XAUUSD',
                 service_fee = 4/100000.0,
                 leverage=1,
                 point_base=10,
                 skip_rows=False):
        """
        初始化函数
        :param time_file:
        :param price_folder:
        :param price_prefix:
        :param service_fee:
        """
        self.time_file=time_file
        self.price_folder=price_folder
        self.price_prefix=price_prefix
        self.service_fee=service_fee
        self.leverage=leverage
        self.point_base=point_base
        self.skip_rows=skip_rows
        self.args = {
            "time_file": self.time_file,
            "time_column": 'time',
            "price_folder": self.price_folder,
            "price_prefix": self.price_prefix,
            "price_column": 'price',
            "leverage": self.leverage,
            "point_base": self.point_base,
            "service_fee": self.service_fee,
            "skip_rows": skip_rows
        }

    def get_train_data_small_dataset(self):
        timefile_list=os.listdir(self.price_folder)
        time_list=[datetime.strptime(i.strip('XAUUSD').strip('.csv').replace('.','/'),'%Y/%m/%d') for i in timefile_list]
        time_list=sorted(time_list)
        data_all=pd.read_csv('../data/input/x/NF.csv')
        data=pd.DataFrame()
        data_all['datetime_format']=data_all['date'].apply(lambda x:datetime.strptime(x,'%Y/%m/%d'))
        data_all=data_all.sort_values(by='datetime_format')
        for i in data_all.datetime_format:
            if i in time_list:
                data=data.append(data_all[data_all.datetime_format==i])
        # data=data_all.copy()
        y_price=get_price_after(5,args=self.args,profit_point = 1000,loss_point = -1000, service_fee = self.service_fee,
                                save_price = False)
        data['year']=data_all['date'].apply(lambda x:x.split('/')[0])
        data['signal']=y_price['signal'].values

        data['NF_AF']=data['NF_actual']-data['NF_forecast']
        data['NF_AP']=data['NF_actual']-data['previous']
        data['NF_FP']=data['NF_forecast']-data['previous']

        data['UR_AF'] = data['UR_actual'] - data['UR_forecast']
        data['UR_AP'] = data['UR_actual'] - data['UR_previous']
        data['UR_FP'] = data['UR_forecast'] - data['UR_previous']

        data['HS_AF'] = data['HS_actual'] - data['HS_forecast']
        data['HS_AP'] = data['HS_actual'] - data['HS_previous']
        data['HS_FP'] = data['HS_forecast'] - data['HS_previous']
        data=data.drop(columns=['direction','turn','datetime_format'])
        return data,['NF_actual', 'NF_forecast', 'previous', 'UR_actual', 'UR_forecast',
       'UR_previous', 'HS_actual', 'HS_forecast', 'HS_previous'],['NF_AF', 'NF_AP', 'NF_FP', 'UR_AF', 'UR_AP', 'UR_FP',
                                                                  'HS_AF', 'HS_AP', 'HS_FP']
    def get_train_data_large_dataset(self):
        hs=pd.read_csv('../data/input/x/HS_normed.csv')
        nf=pd.read_csv('../data/input/x/NF_normed.csv')
        ur=pd.read_csv('../data/input/x/UR_normed.csv')
        res=[hs,nf,ur]
        name=['HS','NF','UR']
        changed_res=[]
        for c,i in enumerate(res):
            i['year']=i['timestamp'].apply(lambda x:int(x.split('/')[0]))
            i['AF_{}'.format(name[c])]=i['actual']-i['forecast']
            i['AP_{}'.format(name[c])]=i['actual']-i['previous']
            i['FP_{}'.format(name[c])]=i['forecast']-i['previous']
            changed_res.append(i[i.year>=2009].drop(columns=['year']))
        res=pd.merge(changed_res[0],changed_res[1],how='inner',on='timestamp',suffixes=('_HS','_NF'))
        changed_res[-1]=changed_res[-1].rename(columns={'actual':'actual_UR','forecast':'forecast_UR','previous':'previous_UR'},inplace=False)
        res=pd.merge(res,changed_res[-1],on='timestamp',how='inner')
        res['datetime']=res['timestamp'].apply(lambda x:datetime.strptime(x,'%Y/%m/%d %H:%M'))
        res=res.sort_values(by='datetime')
        res=res.drop(columns=['timestamp'])
        # res['datetime'].to_csv('NF_large_dataset_timestamp.csv',index=False)
        res.rename(columns={'datetime':'time'},inplace=True)
        original_features=[i for i in res.columns if i.split('_')[0][0] in ['f','p','a']]
        diff_features=[i for i in res.columns if i.split('_')[0][0] in ['F','A']]
        return res,original_features,diff_features

    def train(self,train_num,is_large=False):
        if is_large==False:
            data, original_feature, diff_feature = self.get_train_data_small_dataset()
        else:
            data,original_feature,diff_feature=self.get_train_data_large_dataset()
        all_index=data.index
        train_index=all_index[:train_num]
        test_index=all_index[train_num:]
        x_train=data.loc[train_index,diff_feature]
        y_train=data.loc[train_index,'signal']
        x_test=data.loc[test_index,diff_feature]
        y_test=data.loc[test_index,'signal']
        from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC, LinearSVC,NuSVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        mapping={
            'LR': LogisticRegression(),
            'RC':RidgeClassifier(),
            'SGD':SGDClassifier(),
            'KNN':KNeighborsClassifier(n_neighbors=3),
            'SVC':SVC(),
            'LinearSVC':LinearSVC(),
            'NuSVC':NuSVC(),
            'RF':RandomForestClassifier(),
            'GBC':GradientBoostingClassifier(),
            'ABC':AdaBoostClassifier(),
            'BC':BaggingClassifier(),
            'DTC':DecisionTreeClassifier()
        }
        accs=[]
        models=[]
        preds=[]
        for model_name in mapping:
            m=mapping[model_name]
            m.fit(x_train,y_train)
            pred=m.predict(x_test)
            acc=np.mean(pred==y_test.values)
            accs.append(acc)
            models.append(m)
            preds.append(pred)
            print('{} acc is {}'.format(model_name,acc))
        best_index=np.argmax(accs)
        import joblib
        joblib.dump(models[best_index],'../data/output/model/NF_{}_acc_{}.model'.format(list(mapping.keys())[best_index],
                                                                                        accs[best_index]))
        prediction=pd.DataFrame()
        prediction['time']=data.loc[test_index,'time']
        prediction['signal']=preds[best_index]
        prediction.to_csv('../data/output/predictions/NF_{}_prediction.csv'.format(list(mapping.keys())[best_index]),index=False)
        try:
            print(models[best_index].feature_importances_)
        except Exception as e:
            print(models[7].feature_importances_)

if __name__=='__main__':
    a=NF()
    a.train(train_num=43)




