
#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
# Rule-Based Customer Return Calculation
#############################################

#############################################
# İş Problemi / Business Problem
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

#A game company wants to create new level-based personas using some of their customers' features.
#It wants to create segments according to these new customer definitions and estimate how much the new customers can earn according to these segments.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.
# Example: It is desired to determine how much a 25-year-old male user from Turkey, who is an IOS user, can earn on average.

#############################################
# Veri Seti Hikayesi / Dataset Story
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# The Persona.csv dataset contains the prices of the products sold by an international game company and some demographic information of the users who buy these products.
# The data set consists of records created in each sales transaction. A user with certain demographics may have made more than one purchase.

# Price: Müşterinin harcama tutarı / The customer's spending amount
# Source: Müşterinin bağlandığı cihaz türü / The type of device the customer is connecting to
# Sex: Müşterinin cinsiyeti / Customer's gender
# Country: Müşterinin ülkesi/ Customer's counrty
# Age: Müşterinin yaşı / Customer's age

################# Uygulama Öncesi/ Unedited dataset #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası / The edited version of the dataset #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ / PROJECT TASKS
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız. / Answer the following questions
#############################################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
# Q1: Read persone.csv and show general information about the dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("C:/Users/Lenovo/PycharmProjects/persona.csv")
df.columns

# This function gives us that our dataframe some general information
def demog_inf(dataframe, head=5):
    print("######################## Shape ########################")
    print(dataframe.shape)
    print("######################## Types ########################")
    print(dataframe.dtypes)
    print("######################## Head ########################")
    print(dataframe.head(head))
    print("######################## Tail ########################")
    print(dataframe.tail(head))
    print("######################## NA ########################")
    print(dataframe.isnull().sum())
    print("######################## Quantiles ########################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)  # Sayısal değişkenlerin dağılım bilgisi

demog_inf(df)

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
# Q2: How many unique SOURCE are there? What are their frequencies?

df["SOURCE"].unique()
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır? / Q3: How many unique PRICE are there?

df["PRICE"].unique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş? / Q4: How many sales were made from which PRICE?
df["PRICE"].value_counts().sort_index()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş? / Q5: How many sales from which country?
df[["COUNTRY"]].value_counts()


# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
# Q6: How much was earned in total from sales by country?

df.groupby("COUNTRY").agg({"PRICE": "sum"})


# Soru 7: SOURCE türlerine göre satış sayıları nedir?
# Q7: What are the number of sales according to SOURCE types?

df.groupby("SOURCE").agg({"PRICE": "count"})

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
# Q8: What are the averages of PRICE by countries?

df.groupby("COUNTRY")["PRICE"].mean()

# 2nd way. We can answer the following two question with this target_summary function
def target_summary(dataframe, target, data_col):
    print(pd.DataFrame({"TARGET MEAN": dataframe.groupby(data_col)[target].mean()}))


target_summary(df, "PRICE", "COUNTRY")

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
target_summary(df, "PRICE", "SOURCE")

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
target_summary(df, "PRICE", ["COUNTRY", "SOURCE"])


#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir? / What are the average earnings in breakdown of COUNTRY, SOURCE, SEX, AGE?
#############################################

target_summary(df, "PRICE", ["COUNTRY", "SOURCE", "SEX", "AGE"])

# 2nd way
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız. / Sort the output by PRICE.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

# To better see the output from the previous question, apply the sort_values method to PRICE in descending order.
# Save the output as agg_df. (i did this step in the previous question.)

agg_df.sort_values("PRICE", ascending=False)

#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz. / Convert the names in the index to variable names.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.

# All variables except PRICE in the output of the third question are index names.
# Convert these indexes as variables.

# İpucu: reset_index() / Clue reset_index()
# agg_df.reset_index(inplace=True)

agg_df = agg_df.reset_index(["COUNTRY", "SOURCE", "SEX", "AGE"])


#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz. /  Q5: Convert AGE variable to categorical variable and add it to agg_df.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz. / Convert AGE variable to categorical variable
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz. / Make up the intervals as you think will be persuasive.
# Örneğin:(like that) '0_18', '19_23', '24_30', '31_40', '41_70'


a = [0,18,23,30,40, agg_df["AGE"].max()]
b = ["0_18", "19_23", "24_30", "31_40", "41_" + str(agg_df["AGE"].max())]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=a, labels=b)


agg_df["AGE_CAT"].unique()


#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz. / Define new level based customers and add them as variables to the dataset.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Define a variable called customers_level_based and add this variable to the dataset.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

# Attention!!
# After creating customers level based values with list comprehension, these values need to be deduplicated.
# For example, it could be more than one of the following: USA_ANDROID_MALE_0_18
# It is necessary to get the price averages using the groupby function.


agg_df["CUSTOMER_LEVEL_BASED"] = [(col[0] + "_" + col[1] + "_" + col[2] + "_" + col[5]).upper() for col in agg_df.values]
# 2nd way agg_df["customers_level_based"] = ["_".join([row[0], row[1], row[2], row[5]]).upper() for row in agg_df.values]
# 3rd way agg_df["customers_level_based"] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_CAT']].agg(lambda x: '_'.join(x).upper(), axis=1)
# agg_df[["customers_level_based", "PRICE"]].head()
agg_df.head()
agg_df["CUSTOMER_LEVEL_BASED"].value_counts()  # We can see from some variables that there is more than one.

agg_df = agg_df.groupby("CUSTOMER_LEVEL_BASED").agg({"PRICE": "mean"})
agg_df = agg_df.reset_index()

agg_df["CUSTOMER_LEVEL_BASED"].value_counts()  # We can observe that the variables become singular.
agg_df.head()



#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız. Segment new customers like (USA_ANDROID_MALE_0_18).
#############################################
# PRICE'a göre segmentlere ayırınız. / Segment by PRICE.
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz. / add the segments to the agg_df with the naming "SEGMENT"
# segmentleri betimleyiniz. / describe the segments


agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])  # segment oluşumunda en iyi olan büyük olması olduğu için burada tersten sıralama yaptık
agg_df.head()
agg_df["SEGMENT"].value_counts()
agg_df.groupby(["SEGMENT"]).agg({"PRICE": ["mean", "max", "sum"]})
agg_df["SEGMENT"].describe()
# agg_df.sort_values("PRICE", ascending=False)


#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz. / Classify the new customers and estimate how much income they can bring.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# Which segment does a 33-year-old Turkish woman using ANDROID belong to? How much income is expected on average?
new_customer = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMER_LEVEL_BASED"] == new_customer]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
# In which segment and on average how much income would a 35-year-old French woman using iOS expect to earn?

new_customer = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["CUSTOMER_LEVEL_BASED"] == new_customer]