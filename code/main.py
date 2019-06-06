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
        signal_file=pd.read_csv('../data/input/x/signal_NF_30.csv')
        t=signal_file[signal_file.signal==0].index
        signal_file.drop(index=t,inplace=True)
        res.drop(index=t,inplace=True)
        res['signal']=signal_file['signal']
        print(diff_features)
        print(original_features)
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
        from sklearn.ensemble import ExtraTreesClassifier
        mapping={
            'LR': LogisticRegression(penalty='l2'),
            'RC':RidgeClassifier(),
            'SGD':SGDClassifier(max_iter=2000, tol=1e-3),
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
        joblib.dump(models[best_index],'../data/output/model/NF_{}_{}_acc_{}.model'.format('large' if is_large else '59',list(mapping.keys())[best_index],
                                                                                        accs[best_index]))
        prediction=pd.DataFrame()
        prediction['time']=data.loc[test_index,'time']
        prediction['signal']=preds[best_index]
        prediction.to_csv('../data/output/predictions/NF_{}_{}_prediction.csv'.format('large' if is_large else '59',list(mapping.keys())[best_index]),index=False)
        fs=ExtraTreesClassifier()
        fs.fit(x_train,y_train)
        fi=list(fs.feature_importances_)
        import heapq
        t=map(fi.index,heapq.nlargest(4,fi))
        z = list(t)
        selected_features=[diff_feature[i] for i in z]
        print(selected_features)
        f=open('../data/output/model/selected_features.txt','w',encoding='utf-8')
        f.writelines('\n'.join(selected_features))

    def prediction(self,model_path,selected_features,is_large=True):
        if is_large==False:
            data, original_feature, diff_feature = self.get_train_data_small_dataset()
        else:
            data,original_feature,diff_feature=self.get_train_data_large_dataset()
        sf=open(selected_features,'r',encoding='utf-8').readlines()
        sf=[i.strip('\n') for i in sf]
        import joblib
        model=joblib.load(model_path)
        x=data[diff_feature]
        y=data['signal']
        data['prediction']=model.predict(x)
        mean_x=data[sf].mean()
        res=[]
        for i in data.index:
            temp=str()
            for j in sf:
                if data.loc[i,j]>=mean_x[j]:
                    temp+='2'
                else:
                    temp+='1'
            res.append(temp)
        data['categories']=res
        df=pd.DataFrame()
        df=df.append(mean_x,ignore_index=True)
        df.to_csv('selected_features_mean_value.csv',index=False)
        data.to_csv('NF_LR_prediction.csv',index=False)

def get_signal_categories(input_data,model_path,selected_features_path):
    """
    created by 李帅NLP
    按照这个顺序给输入的数据列表(list)
    ['actual_HS', 'forecast_HS', 'previous_HS', 'actual_NF', 'forecast_NF',
     'previous_NF', 'actual_UR', 'forecast_UR', 'previous_UR']
    :param input_data: 输入的数据
    :param model_path: 预测模型的路径
    :param selected_features_path: 特征选择后的特征表格（来自训练过程）
    :return:signal(1,-1)  category(string)
    """
    if isinstance(input_data,list)==False:
        raise TypeError('输入必须是按照一定固定顺序是列表数据（list)!')
    else:
        if len(input_data)!=9:
            raise TypeError('输入的列表必须有9个值，请检查输入值的个数')
        else:
            import joblib, numpy as np
            import pandas as pd
            model = joblib.load(model_path)
            sf = pd.read_csv(selected_features_path)
            x = []
            for i in range(len(input_data)):
                if i % 3 == 0:
                    x.append(input_data[i] - input_data[i + 1])  # A-F
                    x.append(input_data[i] - input_data[i + 2])  # A-P
                    x.append(input_data[i + 1] - input_data[i + 2])  # F-P
            diff_features = ['AF_HS', 'AP_HS', 'FP_HS', 'AF_NF', 'AP_NF',
                             'FP_NF', 'AF_UR', 'AP_UR', 'FP_UR']
            x_df=pd.DataFrame(columns=diff_features)
            x_df=x_df.append(pd.Series({diff_features[i]:x[i] for i in range(9)}),ignore_index=True)
            print(x_df)
            signal = model.predict(x_df)
            category = str()
            for i in range(len(diff_features)):
                if diff_features[i] in list(sf.columns):
                    if x[i] >= sf.loc[0, diff_features[i]]:
                        category += '2'
                    else:
                        category += '1'
            return list(signal)[0], category

if __name__=='__main__':
    print(get_signal_categories(
        input_data=[0.2,0.3,0.2,-0.09999999999999998,0.0,0.09999999999999998,263,181.0,189.0],
        model_path='../data/output/model/NF_large_LR_acc_0.8823529411764706.model',
        selected_features_path='../data/output/predictions/selected_features_mean_value.csv'
    ))
    # a=NF()
    # a.prediction(model_path='../data/output/model/NF_large_LR_acc_0.8823529411764706.model',
    #              selected_features='../data/output/model/selected_features.txt')




