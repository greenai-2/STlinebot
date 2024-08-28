from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.cache import cache


# 引入 linebot SDK
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage, TextMessage, LocationMessage, ImageMessage, StickerSendMessage, ImageSendMessage, LocationSendMessage


# 建立 linebot classs 進行連線
line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)


@csrf_exempt
def callback(request):
    if (request.method == "POST"):
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode('utf-8')


        # 嘗試解密event
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            return HttpResponseForbidden()
        except LineBotApiError:
            return HttpResponseBadRequest()
       
        for event in events:
            print(event)
            if isinstance(event, MessageEvent):
                # 文字訊息事件
                if isinstance(event.message, TextMessage):
                    res_text = event.message.text
                    # line_bot_api.reply_message(event.reply_token, TextSendMessage(text = res_text))
                    user_id = event.source.user_id
                    # line_bot_api.reply_message(event.reply_token, TextSendMessage(text = res_text))
                    if res_text == "股票預測":
                        cache.set(user_id, 'waiting_for_location', timeout=300)  # 5分鐘後狀態自動過期
                        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請輸入股票代碼(例如:1234，大盤指數請輸入:TW):"))
                    elif cache.get(user_id) == 'waiting_for_location':
                    # 用戶輸入股票代碼後，預測結果
                        num = res_text
                        stock_pred=get_stock_predict(num)
                        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"{res_text}的預測是：\n{stock_pred}"))
                        cache.delete(user_id)  # 清除狀態
                    
                    if res_text == "@我要報到":
                        # 回復貼圖訊息
                        line_bot_api.reply_message(event.reply_token, StickerSendMessage(package_id= 11539, sticker_id= 52114110))
                    elif res_text == "@我的名牌":
                        # 回復圖片訊息
                        img_url = "https://i.imgur.com/DWkp6x7.jpeg"
                        line_bot_api.reply_message(event.reply_token, ImageSendMessage(original_content_url=img_url, preview_image_url=img_url))
                    elif res_text == "@車號登入":
                        # 地點訊息
                        line_bot_api.reply_message(event.reply_token, LocationSendMessage(title = "停車場", address="402台灣台中市南區仁義街119號", latitude=24.12355577054, longitude=120.6732894759063))
                    else:
                        line_bot_api.reply_message(event.reply_token, TextSendMessage(text = res_text))
               
                # 我的位置事件
                if isinstance(event.message, LocationMessage):
                    res_text = "{} {}".format(event.message.latitude, event.message.longitude)
                    print(event)
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text = res_text))
               
                # 圖片事件
                if isinstance(event.message, ImageMessage):
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text = "這是一張圖片事件"))


        return HttpResponse()
    else:
        return HttpResponseBadRequest()


def pushMsg(request, uid,msg):
    line_bot_api.push_message(uid, TextSendMessage(text= msg))
    return HttpResponse()

def get_stock_predict(num):

    if num =="TW":
        num="^TWII"
    else:
        num=str(num)+".TW"
###
# 定義 URL
    stock_code=num
#print(stock_code)
    url = f'https://tw.stock.yahoo.com/quote/{stock_code}'
    import requests 
    from bs4 import BeautifulSoup
# 發送GET請求至目標網站
    response = requests.get(url)

    pri_now=0
# 檢查請求是否成功
    if response.status_code == 200:
    # 解析 HTML 網頁
        soup = BeautifulSoup(response.text, 'html.parser')

    # 查找並提取 <h1> 標籤中的文字
        title_element = soup.find('h1', class_='C($c-link-text) Fw(b) Fz(24px) Mend(8px)')
        title = title_element.get_text(strip=True) if title_element else ''

    # 查找並提取 <span> 標籤中的文字
        span_element = soup.find('span', class_='C($c-icon) Fz(24px) Mend(20px)')
        span_text = span_element.get_text(strip=True) if span_element else ''

    # 查找包含價格詳細信息的 <ul> 元素

        ul_element = soup.find('ul', class_='D(f) Fld(c) Flw(w) H(192px) Mx(-16px)')

    # 確保找到 <ul> 元素
        #import requests as rt
        #from bs4 import BeautifulSoup
        if ul_element:
        # 提取成交、開盤、最高、最低價格
            prices = {}
            for li in ul_element.find_all('li', class_='price-detail-item'):
                label = li.find('span', class_='C(#232a31)').get_text(strip=True)
                value = li.find('span', class_='Fw(600)').get_text(strip=True)
                if label in ['成交', '開盤', '最高', '最低']:
                    prices[label] = value
                    pri_now=prices.get('成交', 'N/A')
        # 輸出提取的價格
            #print(f"{title} {span_text}  日期：{today_str}, 時間：{current_time_str}  ")
            #print(f"成交價: {prices.get('成交', 'N/A')}, 開盤價: {prices.get('開盤', 'N/A')}, 最高價: {prices.get('最高', 'N/A')}, 最低價: {prices.get('最低', 'N/A')}")

            #else:
            #print('未找到價格詳細信息的 <ul> 元素。')
        #else:
        #print(f'網頁加載失敗，狀態碼: {response.status_code}')




###
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    data = yf.download(num, start="2023-08-10")
    data = pd.DataFrame(data)

        # Check the actual column names in your DataFrame


        # Assuming the date column is named 'Date', convert it to datetime
    data['Date'] = pd.to_datetime(data.index) #The date is actually in the index
    #data.reset_index(inplace=True) #Reset the index to make 'Date' a column
    data.set_index('Date', inplace=True)

    # 新增一個欄位 OPENfront，其值為前一天的 Open 值
    data['OPEN-1'] = data['Open'].shift(1)
    data['High-1'] = data['High'].shift(1)
    data['Low-1'] = data['Low'].shift(1)
    data['Close-1'] = data['Close'].shift(1)
    data = data.fillna(0)

    # 將所有列轉換為數值型態,無法轉換的保持原樣
    data = data.apply(pd.to_numeric, errors='ignore')
    # 填充数值型列的缺失值，使用每列的均值填充
    data_filled = data.fillna(data.mean(numeric_only=True))

    # 填充字符串类型或非数值型列的缺失值，使用前一个值填充
    dara_filled = data_filled.fillna(method='ffill')

    # 如果前一个值也为空，可以使用后一个值进行填充
    data_filled = data_filled.fillna(method='bfill')

    # 再次检查是否还有缺失值
    missing_data_summary = data_filled.isnull().sum()

    # 返回处理后的数据以及缺失值概况
    missing_data_summary, data_filled.head()


    # # 獲取最後5筆資料
    # last_5_rows = data.tail(5)

    # # 顯示最後5筆資料
    # print(last_5_rows)
    a=data.last_valid_index()
    #date=data.loc[a,'Date']
    #print(date)
    #date_obj = datetime.strptime(date, '%Y/%m/%d')

    date_obj = datetime.strptime(str(a)[:10], '%Y-%m-%d') # Convert to datetime object
    data
    #print(date_obj)
    #data.loc[date_obj]["Close"]

    from datetime import date
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    cp_front=int(data.loc[date_obj]["Close"])
    op_front=int(data.loc[date_obj]["Open"])
    max_p=cp_front*1.09
    min_p=cp_front*0.91
    pri=cp_front
    # 加载数据
    #file_path = '/content/output1.xlsx'
    #data = pd.read_excel(file_path)

    # 选择需要的列
    features = data.columns.difference(['Date', 'Open', 'High', 'Low', 'Close'])
    target = ['Open', 'High', 'Low', 'Close']
    #target = [ '收盤價(元)']
    # 只保留數值型特徵
    numeric_features = data[features].select_dtypes(include=[np.number])

    # 检查和处理缺失值
    numeric_features = numeric_features.fillna(numeric_features.mean())
    data[target] = data[target].fillna(data[target].mean())

    # 数据预处理
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler_features.fit_transform(numeric_features)
    scaled_target = scaler_target.fit_transform(data[target])

    # 创建训练和测试集
    look_back = 10

    def create_dataset(features, target, look_back=1):
        X, y = [], []
        for i in range(len(features) - look_back):
            X.append(features[i:(i + look_back)])
            y.append(target[i + look_back])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, scaled_target, look_back)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
    model.add(LSTM(50))
    #model.add(LSTM(50), input_shape=(look_back, X_train.shape[2]))
    model.add(Dense(4))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型
    model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=0)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 检查预测值中的NaN
    if np.isnan(y_train_pred).any() or np.isnan(y_test_pred).any():
        print("预测值中含有NaN，请检查数据预处理步骤。")

    # 反归一化预测值
    y_train_pred = scaler_target.inverse_transform(y_train_pred)
    y_test_pred = scaler_target.inverse_transform(y_test_pred)
    y_train = scaler_target.inverse_transform(y_train)
    y_test = scaler_target.inverse_transform(y_test)

    # 计算均方误差(MSE)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # 计算平均绝对百分比误差(MAPE)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

    # print(f'Train MSE: {train_mse}')
    # print(f'Test MSE: {test_mse}')
    # print(f'Train MAPE: {train_mape}')
    # print(f'Test MAPE: {test_mape}')

    # 特征重要性分析
    importances = model.layers[0].get_weights()[0].sum(axis=1)
    importances /= importances.sum()
    feature_importance = {feature: importance for feature, importance in zip(numeric_features.columns, importances)}
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    #print("Feature Importances:")
    #for feature, importance in sorted_importance.items():
        #print(f'{feature}: {importance:.4f}')
    #model.evaluate(X_test, y_test)
    #print(f'Train MSE: {train_mse}')
    #print(f'Test MSE: {test_mse}')
    #print(f'Train MAPE: {train_mape}')
    #print(f'Test MAPE: {test_mape}')
    Lop=y_test_pred[0,0]
    Lhp=y_test_pred[0,1]
    Llp=y_test_pred[0,2]
    Lcp=y_test_pred[0,3]
    La=data.last_valid_index()
    #date=data.loc[a,'Date']
    # r=abs((cp_front-op_front)/(cp-op))
    # print(f"r is {r}")
    # op=op*r
    # hp=hp*r
    # lp=lp*r
    # cp=cp*r
    #print(date)
    #print(f"Day : 開盤價: {op:.2f}, 最高價: {hp:.2f}, 最低價: {lp:.2f}, 收盤價: {cp:.2f}")
    #for i, (op, hp, lp, cp) in enumerate(zip(y_test_pred[:, 0], y_test_pred[:, 1], y_test_pred[:, 2], y_test_pred[:, 3])):
       #print(f"Day : 開盤價: {op:.2f}, 最高價: {hp:.2f}, 最低價: {lp:.2f}, 收盤價: {cp:.2f}")
        # if op<lp:
        #   lp=op
        # elif cp<lp:
        #   lp=cp
        # if op>hp:
        #  hp=op
        # elif cp>hp:
        #    hp=cp
    next_day = date_obj + timedelta(days= 1)

    # 將結果格式化為所需的字符串格式
    next_day_str = next_day.strftime('%Y/%m/%d')
        #print(f"{next_day_str} : 開盤價: {op:.2f}, 最高價: {hp:.2f}, 最低價: {lp:.2f}, 收盤價: {cp:.2f}")
    Lr=abs((pri)/(Lop))
    #print(f"r is {Lr}")
    Lop=Lop*Lr
    Lhp=Lhp*Lr
    Llp=Llp*Lr
    Lcp=Lcp*Lr
    #for i, (op, hp, lp, cp) in enumerate(zip(y_test_pred[:, 0], y_test_pred[:, 1], y_test_pred[:, 2], y_test_pred[:, 3])):
    # if op<lp:
    #    lp=op
    # elif cp<lp:
    #    lp=cp
    # if op>hp:
    #    hp=op
    # elif cp>hp:
    #    hp=cp
    Llp=min(Lop, Lhp, Llp, Lcp)*0.99
    Lhp=max(Lop, Lhp, Llp, Lcp)*1.01
    if Lhp>max_p:
      Lcp=Lcp-(Lhp-max_p)
      Lhp=max_p

    if Llp<min_p:
      Lcp=rcp+(min_p-Llp)
      Llp=min_p
    lstm=f"{num}_{next_day_str}_LSTM: 開盤價: {Lop:.2f}, 最高價: {Lhp:.2f}, 最低價: {Llp:.2f}, 收盤價: {Lcp:.2f}"

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import yfinance as yf
    from datetime import timedelta
    #del min  # 刪除名為 min 的變數
    # 讀取數據
    #df = pd.read_csv('your_data.csv')  # 請替換為您的數據文件名
    pri=cp_front
    # 將日期轉換為時間特徵
    # data['日期'] = pd.to_datetime(data['年月日'])
    # data['年'] = data['日期'].dt.year
    # data['月'] = data['日期'].dt.month
    # data['日'] = data['日期'].dt.day
    data = yf.download(num, start="2000-01-01")
    data = pd.DataFrame(data)
    data['OPEN-1'] = data['Open'].shift(1)
    data['High-1'] = data['High'].shift(1)
    data['Low-1'] = data['Low'].shift(1)
    data['Close-1'] = data['Close'].shift(1)
    data = data.fillna(0)
    cp_front=int(data.loc[date_obj]["Close"])
    op_front=int(data.loc[date_obj]["Open"])
    #print(f"cp_front is {cp_front}")
    #print(f"op_front is {op_front}")
    max_p=cp_front*1.09
    min_p=cp_front*0.91

    # 定義特徵和目標變量
    feature = [col for col in data.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close']]
    target = [ 'Open', 'High', 'Low', 'Close']
    #target = ['開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)']
    X = data[feature]
    y = data[target]

    #features = data.columns.difference(['年月日', '開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)'])
    #target = ['開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)']
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
    #target = [ '收盤價(元)']
    # 只保留數值型特徵
    numeric_features = data[feature].select_dtypes(include=[np.number])

    # 检查和处理缺失值
    numeric_features = numeric_features.fillna(numeric_features.mean())
    data[target] = data[target].fillna(data[target].mean())

    # 数据预处理
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler_features.fit_transform(numeric_features)
    scaled_target = scaler_target.fit_transform(data[target])

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.25, random_state=100)

    # 創建並訓練多輸出隨機森林模型
    rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=100,max_depth=5))

    rf_model.fit(X_train, y_train)

    # 預測
    y_pred = rf_model.predict(X_test)

    # 評估模型
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    #print("均方誤差 (MSE):")
    for i, col in enumerate(target):
        print(f"{col}: {mse[i]:.4f}")

    #print("\nR2 分數:")
    for i, col in enumerate(target):
        print(f"{col}: {r2[i]:.4f}")

    # 計算特徵重要性
    feature_importance = np.mean([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

    # 將特徵重要性與特徵名稱配對並排序
    feature_importance_paired = sorted(zip(feature, feature_importance), key=lambda x: x[1], reverse=True)

    #print("\n特徵重要性:")
    #for feature, importance in feature_importance_paired:
         #print(f"{feature}: {importance:.4f}")
    # 繪製特徵重要性條形圖
    features, importances = zip(*feature_importance_paired)
    #plt.figure(figsize=(10, 6))
    #plt.barh(features, importances, color='skyblue')
    #plt.xlabel("rate")
    #plt.ylabel("features")
    #plt.title("features effects")
    #plt.gca().invert_yaxis()  # 使特徵按重要性由高到低顯示
    #plt.show()
    #調整結果
    rop=y_pred[0,0]*1
    rhp=y_pred[0,1]*1
    rlp=y_pred[0,2]*1
    #print(f"lp is {lp}")
    rcp=y_pred[0,3]*1
    rr=abs((pri)/(rop))
    #print(f"r is {rr}")
    rop=rop*rr
    rhp=rhp*rr
    rlp=rlp*rr
    rcp=rcp*rr
    #for i, (op, hp, lp, cp) in enumerate(zip(y_test_pred[:, 0], y_test_pred[:, 1], y_test_pred[:, 2], y_test_pred[:, 3])):
       # print(f"Day : 開盤價: {op:.2f}, 最高價: {hp:.2f}, 最低價: {lp:.2f}, 收盤價: {cp:.2f}")
    #a=data.last_valid_index()
    #date=data.loc[a,'Date']
    #for i, (op, hp, lp, cp) in enumerate(zip(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]),start=1):

    # if op<lp:
    #   lp=op
    # elif cp<lp:
    #   lp=cp
    # if op>hp:
    #   hp=op
    # elif cp>hp:
    #   hp=cp
    rlp= min(rop, rhp, rlp, rcp)*0.99
    rhp= max(rop, rhp, rlp, rcp)*1.01

    if rhp>max_p:
      rcp=rcp-(rhp-max_p)
      rhp=max_p



    if rlp<min_p:
      rcp=rcp+(min_p-rlp)
      rlp=min_p



    #print(f"{date_obj}: 開盤價: {op:.2f}, 最高價: {hp:.2f}, 最低價: {lp:.2f}, 收盤價: {cp:.2f}")
    next_day = date_obj + timedelta(days=1)

    # 將結果格式化為所需的字符串格式
    next_day_str = next_day.strftime('%Y/%m/%d')
    rodf=f"{num}_{next_day_str}_隨機森林: 開盤價: {rop:.2f}, 最高價: {rhp:.2f}, 最低價: {rlp:.2f}, 收盤價: {rcp:.2f}"

    #print(abs(cp_front-rop))
    #print(abs(cp_front-Lop))
    if abs(cp_front-rop)>abs(cp_front-Lop):
      op=Lop
      hp=Lhp
      lp=Llp
      cp=Lcp
    else:
      op=rop
      hp=rhp
      lp=rlp
      cp=rcp
    

    import matplotlib.pyplot as plt
    import pandas as pd
    data = yf.download(num, start="2024-07-01")
    data = pd.DataFrame(data)
    #假设您的数据已经加载到DataFrame中，并且包含名为 'Date' 的日期列
    #data['Date'] = pd.to_datetime(data['Date'])  # 确保 'Date' 列为日期时间类型
    #start_date = '2024-07-01'
    #filtered_data = data[data['Date'] >= start_date]  # 过滤数据
    #plt.figure(figsize=(15, 10))
    #for column in target:
    #    plt.plot(data.index, data[column], label=column)
#
    #plt.title('price-time')
    #plt.xlabel('Date')
    #plt.ylabel('price')
    #plt.legend()
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.show()

    # 繪製每個欄位的單獨折線圖
    #fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    #fig.suptitle('price-time')

    #for i, column in enumerate(target):
        #row = i // 2
        #col = i % 2
        #plt.title(column)
        #axs[row, col].plot(data.index, data[column])
        #axs[row, col].set_title(column)
        #axs[row, col].set_xlabel('Date')
        #axs[row, col].set_ylabel('price)')
        #axs[row, col].tick_params(axis='x', rotation=45)

    #plt.tight_layout()
    #plt.show()
    from datetime import datetime, timedelta
    data = yf.download(num, start="2000-08-10")#end="")
    data = pd.DataFrame(data)
    import datetime
    nowtime=datetime.datetime.now()+datetime.timedelta(hours=0)
    nowtime=nowtime.strftime("%H:%M:%S")
    if "00:00"<=nowtime<"13:30":
        next_day = date_obj + timedelta(days= 0)
    else:
        next_day = date_obj + timedelta(days= 1)
    next_day_str = next_day.strftime('%Y/%m/%d')
    ##
    if isinstance(pri_now, str):
        try:
            pri_now = float(pri_now.replace(',', '')) # Remove commas before conversion
        except ValueError:
            pri_now = 0  # Or handle the error as you see fit
    #print(f"{num}_{next_day_str}_Prediction: 現時價:{pri_now:.2f},開盤價: {op:.2f}, 最高價: {hp:.2f}, 最低價: {lp:.2f}, 收盤價: {cp:.2f}")
    return f"{num}_{next_day_str}_Prediction: 現時價:{pri_now:.2f},開盤價: {op:.2f}, 最高價: {hp:.2f}, 最低價: {lp:.2f}, 收盤價: {cp:.2f}"
    #