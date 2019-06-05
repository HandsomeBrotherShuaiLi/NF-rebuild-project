import os
import pandas as pd
import numpy as np
from datetime import datetime,timedelta

def get_price_after(second,
                    time_file = '../data/input/statement+implement.csv',
                    time_column = 'time(UTC)',
                    price_folder = '../data/input/XAUUSD_FOMC_30min/',
                    price_prefix = 'XAUUSD',
                    price_column = 'price',
                    profit_point = 100000,
                    loss_point = -100000,
                    service_fee = 4/100000.0,
                    args = None,
                    save_price = False):

    if args != None:
        time_file = args['time_file']
        time_column = args['time_column']
        price_folder = args['price_folder']
        price_prefix = args['price_prefix']
        price_column = args['price_column']
        point_base = args['point_base']
        skip_rows = args['skip_rows']

    test = pd.read_csv(time_file)
    test.index=pd.to_datetime(test[time_column])
    time = []
    signal = []

    if save_price: print('saving file to data/output/price_temp')
    
    for i in test.index:
        try:
            date = str(i)[:4]+'.'+str(i)[5:7]+'.'+str(i)[8:10]
            begin = (int(str(i)[10:13])*36000 + int(str(i)[14:16])*600)

            if skip_rows: skip_rows = [i for i in range(1,begin+1)]

            data = pd.DataFrame(pd.read_csv((price_folder+price_prefix+date+'.csv'),nrows = (10*second + 2),skiprows = skip_rows))
            if save_price:
                print(data.shape)
                data.to_csv('../data/output/price_temp/'+price_prefix+date+'.csv')


            data.index = pd.to_datetime(data.index)
            openTime = data.index[0]

            price = data[price_column][0]
            openPrice = price
            
            flag = 0
            max_long = 0
            max_short = 0
            
            for j in range(1,second*10):
                
                currentPrice = data[price_column][j]
                
                if currentPrice - openPrice > max_long:
                    max_long = currentPrice - openPrice
                
                if openPrice - currentPrice > max_short:
                    max_short = openPrice - currentPrice
                    

                if (currentPrice-openPrice > (profit_point/point_base + service_fee)) and (max_short < (-loss_point/point_base - service_fee)):
                    time.append(openTime)
                    signal.append(1)
                    flag = 1
                    break
                    
                elif (openPrice - currentPrice > (profit_point/point_base + service_fee)) and (max_long < (-loss_point/point_base - service_fee)):
                    time.append(openTime)
                    signal.append(-1)
                    flag = 1
                    break
            
            if flag == 0:
                
                time.append(openTime)
                currentPrice = data[price_column][second*10]
                
                if (max_long > max_short):
                    signal.append(1)
                elif (max_long < max_short):
                    signal.append(-1)
                else:
                    if (currentPrice > openPrice):
                        signal.append(1)
                    elif (currentPrice < openPrice):
                        signal.append(-1)
                    else:
                        second_temp = second
                        signal_append_flag = 0
                        while (second_temp/2.0 >1) and (signal_append_flag == 0):
                            second_temp = second_temp/2.0
                            currentPrice = data[price_column][int(second_temp*10)]
                            if (currentPrice > openPrice):
                                signal.append(1)
                                signal_append_flag = 1
                            elif (currentPrice < openPrice):
                                signal.append(-1)
                                signal_append_flag = 1

                        if signal_append_flag == 0: 
                            print("price not change, signing 0 at ",str(date))
                            signal.append(0)
            
        except:
            pass
    result = pd.DataFrame(signal,index=time,columns=['signal'])
    result.index.name='time'
    return result