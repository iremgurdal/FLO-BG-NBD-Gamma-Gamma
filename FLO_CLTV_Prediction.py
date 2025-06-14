##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################


###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin
# gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
           # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# BONUS: Tüm süreci fonksiyonlaştırınız.

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

#pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################


###############################################################
# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

df_ = pd.read_csv('0_Cases_and_Projects/FLO_CLTV_Prediction/flo_data_20k.csv')
df = df_.copy()
df.head()

###############################################################
# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

# Aykırı değer eşiklerini hesaplama
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    # tam sayı olması gereken değişkenler için yuvarlama
    if dataframe[variable].dtype in ['int64', 'float64'] and 'freq' in variable:
        low_limit = round(low_limit)
        up_limit = round(up_limit)

    return low_limit, up_limit


# Aykırı değerleri eşiklerle baskılama
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

###############################################################
# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.
df.info()
df.describe()

# Aykırı değer kontrolü yapılacak değişkenler
variables_to_check = [
    'order_num_total_ever_online',
    'order_num_total_ever_offline',
    'customer_value_total_ever_offline',
    'customer_value_total_ever_online'
]

# Aykırı değerleri baskıla
for var in variables_to_check:
    replace_with_thresholds(df, var)


outlier_summary = []

for var in variables_to_check:
    lower, upper = outlier_thresholds(df, var)
    outlier_count = df[(df[var] < lower) | (df[var] > upper)].shape[0]
    outlier_summary.append((var, lower, upper, outlier_count))

    # Boxplot çizimi
    plt.figure()
    df.boxplot(column=var)
    plt.title(f'{var} - Boxplot')
    plt.show()

outlier_summary_df = pd.DataFrame(outlier_summary, columns=['variable', 'low_limit', 'up_limit', 'outlier_count'])



###############################################################
# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['customer_value_total'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

###############################################################
# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.info()

date_columns = ['first_order_date', 'last_order_date', 'last_order_date_online', 'last_order_date_offline']

for col in date_columns:
    df[col] = pd.to_datetime(df[col])



###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################


###############################################################
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

today_date = df['last_order_date'].max() + dt.timedelta(days=2)


###############################################################
# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.



cltv_df = pd.DataFrame()
cltv_df['customer_id'] = df['master_id']
cltv_df['frequency'] = df['order_num_total']
cltv_df['monetary_cltv_avg'] = df['customer_value_total'] / df['order_num_total']
cltv_df['recency_cltv_weekly'] = (df['last_order_date'] - df['first_order_date']).dt.days / 7
cltv_df['T_weekly'] = (today_date - df['first_order_date']).dt.days / 7



###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################


###############################################################
# 1. BG/NBD modelini kurunuz.
# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.

# BG/NBD modeli
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])

# 3. ve 6. ay satın alma tahminleri
cltv_df['exp_sales_3_month'] = bgf.predict(12, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])
cltv_df['exp_sales_6_month'] = bgf.predict(24, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# 3 ayda en çok satın alma yapacak 10 kişi
top_3_month = cltv_df.sort_values('exp_sales_3_month', ascending=False).head(10)

# 6 ayda en çok satın alma yapacak 10 kişi
top_6_month = cltv_df.sort_values('exp_sales_6_month', ascending=False).head(10)


###############################################################
# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip
# exp_average_value olarak cltv dataframe'ine ekleyiniz.

cltv_df = cltv_df[cltv_df['frequency'] > 1]

cltv_df['frequency'] = cltv_df['frequency'].astype(int)

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

# Tahmini ortalama harcamalar
cltv_df['exp_average_value'] = ggf.conditional_expected_average_profit(
    cltv_df['frequency'], cltv_df['monetary_cltv_avg']
)

###############################################################
# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df['cltv'] = ggf.customer_lifetime_value(
    transaction_prediction_model=bgf,
    frequency=cltv_df['frequency'],
    recency=cltv_df['recency_cltv_weekly'],
    T=cltv_df['T_weekly'],
    monetary_value=cltv_df['monetary_cltv_avg'],
    time=6,  # ay
    freq='W',  # zaman birimi: hafta
    discount_rate=0.01
)

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

top_20_cltv = cltv_df.sort_values('cltv', ascending=False).head(20)



###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################


###############################################################
# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_df['cltv_segment'] = pd.qcut(cltv_df['cltv'], 4, labels=['D', 'C', 'B', 'A'])

###############################################################
# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

segment_means = cltv_df.groupby('cltv_segment').agg({
    'recency_cltv_weekly': 'mean',
    'frequency': 'mean',
    'monetary_cltv_avg': 'mean',
    'cltv': 'mean'
}).round(2)








