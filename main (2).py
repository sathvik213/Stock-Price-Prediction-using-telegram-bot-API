import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance
import  datetime

# ticker_symbol='KO'
def run(ticker_symbol):

    startDate='2021-01-01'
    endDate=datetime.datetime.now()
    Data=yfinance.Ticker(ticker_symbol)
    Data=Data.history(start=startDate,end=endDate)

    Data.to_csv('new.xlsx')
    new_data=pd.read_csv('new.xlsx')
    SMA_20=pd.DataFrame()#creating empty data frame
    SMA_20['Close']=new_data['Close'].rolling(window=20).mean()
    SMA_20['Close']=SMA_20['Close'].fillna(0)
    # print(SMA_20.head())
    send_msg('bot started running .....')
    send_msg('Initialising...')
    a=SMA_20['Close']
    # print(a)
    send_msg('recent close stock price of ticker '+ticker_symbol+str(a))
    plt.plot(new_data['Date'],new_data['Close'],color='green')
    plt.savefig("close_graph.jpg")
    send_msg('Recent one year(aprox.) close price graph for company with ticker '+ticker_symbol)
    sendImage('close_graph.jpg')
    plt.plot(range(len(SMA_20['Close'])),SMA_20['Close'],color='red')
    plt.savefig('SMA20_graph.jpg')
    send_msg('Recent one year SMA20 close price graph for company with ticker '+ticker_symbol+' with red graph as SMA20')
    sendImage('SMA20_graph.jpg')
    plt.close()
def scatter(ticker_symbol):
    startDate = '2021-01-01'
    endDate = datetime.datetime.now()
    Data = yfinance.Ticker(ticker_symbol)
    Data = Data.history(start=startDate, end=endDate)

    Data.to_csv('new.xlsx')
    new_data = pd.read_csv('new.xlsx')
    p=len(new_data['Date'])
    reg_data=pd.DataFrame(columns=['Close'])
    reg_data=new_data['Close']
    # print(reg_data[p-30:p])
    reg_final_list=reg_data[p-30:p]
    send_final=list(reg_final_list)
    plt.scatter(range(len(reg_final_list)),reg_final_list)
    plt.savefig('Scatter_data.jpg')
    send_msg('Scatter plot data for past 30 days ')
    sendImage('Scatter_data.jpg')
    plt.close()
    send_msg('Applying linear regression ML algo for prediction purpose....')
    time.sleep(1)
    list_of_tuples=list(zip(range(30),send_final))
    dataset=pd.DataFrame(list_of_tuples,columns=['last_30_days', 'Close'])
    print(dataset)

    # #ML starts

    X=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    from sklearn.linear_model import LinearRegression
    regressor=LinearRegression()
    regressor.fit(X_train,y_train)#here fit() is one of the method of LinearRegression Class
    y_pred=regressor.predict(X_test)#gives predicted values by trained model
    ## print(y_pred)


    ##----------------close price vs last_30_days (training set)----------------
    plt.scatter(X_train,y_train,color='red')
    plt.plot(X_train,regressor.predict(X_train),color='blue')
    plt.title('close price vs last_30_days (training set)')
    plt.xlabel('last_30_days')
    plt.ylabel('close price')
    plt.savefig('close price vs last_30_days (training set).jpg')
    send_msg('\n----------------------------------\nshowing graph related to analysis and prediction regarding \n"close price vs last_30_days (training set)"\n By using model fit line by Machine Learning algorithm linear regression\n----------------------------------\n ')
    sendImage('close price vs last_30_days (training set).jpg')
    plt.close()
    time.sleep(1)
    # #plt.show()


    ##----------------close price vs last_30_days (test set) observed data-------------------------------
    plt.scatter(X_test,y_test,color='green')
    plt.plot(X_train,regressor.predict(X_train),color='yellow')
    plt.title(' close price vs last_30_days (test set) observed data')
    plt.xlabel('last_30_days')
    plt.ylabel('close price')
    plt.savefig('close price vs last_30_days (test set) observed data.jpg')
    send_msg('\n----------------------------------\nshowing graph related to analysis and prediction regarding \n"close price vs last_30_days (test set) observed data"\n By using model fit line by Machine Learning algorithm linear regression\n----------------------------------\n ')
    sendImage('close price vs last_30_days (test set) observed data.jpg')
    # #plt.show()
    time.sleep(1)


    ##----------------close price vs last_30_days (test set) model data-------------------------------
    plt.scatter(X_test,y_pred,color='red')
    plt.plot(X_train,regressor.predict(X_train),color='blue')
    plt.title('close price vs last_30_days (test set) model data')
    plt.xlabel('last_30_days')
    plt.ylabel('close price')
    plt.savefig('close price vs last_30_days (test set) model data.jpg')
    send_msg('\n----------------------------------\nshowing graph related to analysis and prediction regarding \n"close price vs last_30_days (test set) model data"\n By using model fit line by Machine Learning algorithm linear regression\n----------------------------------\n ')
    sendImage('close price vs last_30_days (test set) model data.jpg')
    time.sleep(1)

    # #plt.show()


    ##-----------------Backend Mathematical Analysis part-------------------------------------------
    ## line_plot = dataset.plot.line(x='last_30_days', y='Close')
    ## plt.show()
    mean_last_30_days = sum(dataset['last_30_days']) / float(len(dataset['last_30_days']))
    print('mean_last_30_days ',mean_last_30_days)
    mean_close = sum(dataset['Close']) / float(len(dataset['Close']))
    print('mean_close ',mean_close)
    # Calculate the variance
    def variance(values, mean):
        return sum([(val-mean)**2 for val in values])
    # Calculate covariance between Experience and Salary
    def covariance(last_30_days,mean_last_30_days , close , mean_close):
        covariance = 0.0
        for r in range(len(last_30_days)):
            covariance = covariance + (last_30_days[r] - mean_last_30_days) * (close[r] - mean_close)
        return covariance

    variance_last_30_days, variance_close = variance(dataset['last_30_days'], mean_last_30_days), variance(dataset['Close'], mean_close)
    print('variance_last_30_days ',variance_last_30_days)
    print('variance_close ',variance_close)

    covariance_l30_close = covariance(dataset['last_30_days'],mean_last_30_days,dataset['Close'],mean_close)
    print('covariance_l30_close ',covariance_l30_close)
    m = covariance_l30_close/ variance_last_30_days
    c = mean_close- m * mean_last_30_days
    x=31
    print(m*x+c)
    pred_value_for_next_day=m*x+c
    send_msg('So final prediction possibility determined by Investar is :\n'+str(pred_value_for_next_day)+'\n\nNOTE:TRADING COMMODITY FUTURES, OPTIONS, CFD’s SPREAD BETTING AND FOREIGN EXCHANGE ("FOREX") INVOLVES HIGH RISKS  ')








#----------------------------------------------------------------------------------------------------------------------------------------------------------------
import telebot
import io
import telegram.ext
import telegram_send
token='5526476876:AAG977Ez0Fg7xajbtt_IYO1UL6N4T26HTP0' #gyanesh
chat_id='-652505017'#gyanesh group
# token='5409994509:AAE6UCISSah7xT3IPtV7W3CzZL9CFa6vYH0'
# chat_id='5409994509'
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
import requests
def send_msg(msg):
    response=requests.post('https://api.telegram.org/bot' + token + '/sendMessage?chat_id=' + chat_id + '&parse_mode=Markdown&text=' + msg)
def sendImage(msg):
    url = "https://api.telegram.org/bot"+token+"/sendPhoto";
    files = {'photo': open(msg, 'rb')}
    data = {'chat_id' : chat_id}
    r= requests.post(url, files=files, data=data)
    print(r.status_code, r.reason, r.content)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
bot=telebot.TeleBot(token)
updater=telegram.ext.Updater(token,use_context=True)
dispatch=updater.dispatcher

def activate(update,context):
    update.message.reply_text('Hello!! Welcome to Investar \n A handy helper to stock analysis')
def help(update,context):
    update.message.reply_text('''
    The following topics are keenly focused in this feature:
    1. ticker symbol
    2. data visualization 
    3. basic data analysis 
    4. data analysis and prediction testing using ML algo 
    5. prediction for next day 
    
    get started by clicking this:
    /deploy
    ''')
def deploy(update,context):
    update.message.reply_text('''As we are still in initial phases please select desired company ticker from below: 
    1. KO(/Cococola)
    2. F(/Ford_motors)
    3. TSLA(/Tesla)
    4. AAPL(/Apple)
    5. AMZN(/Amazon)
    ''')
def Cococola(update,context):
    ticker_symbol='KO'
    run(ticker_symbol)
    scatter(ticker_symbol)
def Ford_motors(update,context):
    ticker_symbol='F'
    run(ticker_symbol)
    scatter(ticker_symbol)
def Tesla(update,context):
    ticker_symbol='TSLA'
    run(ticker_symbol)
    scatter(ticker_symbol)
def Apple(update,context):
    ticker_symbol='AAPL'
    run(ticker_symbol)
    scatter(ticker_symbol)

def Amazon(update, context):
    ticker_symbol = 'AMZN'
    run(ticker_symbol)
    scatter(ticker_symbol)

def handle_message(update,context):
    update.message.reply_text(f'you entered {update.message.text}')

dispatch.add_handler(telegram.ext.CommandHandler('activate',activate))
dispatch.add_handler(telegram.ext.CommandHandler('help',help))
dispatch.add_handler(telegram.ext.CommandHandler('deploy',deploy))
dispatch.add_handler(telegram.ext.CommandHandler('Cococola',Cococola))
dispatch.add_handler(telegram.ext.CommandHandler('Ford_motors',Ford_motors))
dispatch.add_handler(telegram.ext.CommandHandler('Tesla',Tesla))
dispatch.add_handler(telegram.ext.CommandHandler('Amazon',Amazon))
dispatch.add_handler(telegram.ext.CommandHandler('Apple',Apple))
updater.start_polling()
updater.idle()
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------





# def echo(update, context):
#     context.bot.send_message(chat_id=chat_id, text=update.message.text)
#     print(update.message.text)
#
# from telegram.ext import MessageHandler, Filters
# echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)
# dispatch.add_handler(echo_handler)


# sendImage()
# def sendImageRemoteFile(img_url):
#     url = "https://api.telegram.org/bot"+token+"/sendPhoto";
#     remote_image = requests.get(img_url,'rb')
#     photo = io.BytesIO(remote_image.content)
#     photo.name = 'img.png'
#     files = {'photo': photo}
#     data = {'chat_id' : chat_id}
#     r= requests.post(url, files=files, data=data)
#     print(r.status_code, r.reason, r.content)