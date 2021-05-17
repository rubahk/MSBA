#To import the libraries:
import pandas as pd 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from io import StringIO
from PIL import Image
import seaborn as sns
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import squarify


#To read the dataset:
orig_url= 'https://drive.google.com/file/d/1SbCQWs-5Cd-sS3Ie3DvzDdVE-tkrY67B/view?usp=sharing'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
data= pd.read_csv(csv_raw)
print(data.head())

url= "https://drive.google.com/file/d/1aeatvjF8SEI2JAzc9mI_Or5-AakY2Uxy/view?usp=sharing"
image='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]


#inserting the aub logo:
left_col=st.image(image,width=300)


#To Add a title:
st.title("Supermarket Sales in Myanmar")
st.write("by Ruba Al Hakeem|May 2021")
st.set_option('deprecation.showPyplotGlobalUse', False)


#To know the total sales per branches: Question 1
st.subheader('Total sales per branch')
if st.checkbox('Branch A'):
    st.write('The total sales of this branch is 106200.37 $')
if st.checkbox('Branch B'):
    st.write('The total sales of this branch is 106197.67 $')
if st.checkbox('Branch C'):
    st.write('The total sales of this branch is 110568.71 $')


       
#To know the total sales per product line: Question 2
st.subheader('Total sales per product line')
product= data['Product line'].unique()
option = st.selectbox(
    'Which product line do you want to choose?',
     product)

if "Health and beauty" in option:
    st.write('The total sales of this product line is 49193.74 $')
if "Electronic accessories" in option:
    st.write('The total sales of this product line is 54337.5 $')
if "Home and lifestyle" in option:
    st.write('The total sales of this product line is 53861.9 $')
if "Sports and travel" in option:
    st.write('The total sales of this product line is 55122.8 $')
if "Food and beverages" in option:
    st.write('The total sales of this product line is 56144.8 $')
if "Fashion accessories" in option:
    st.write('The total sales of this product line is 54305.9 $')
    

st.text("")

#To construct a pie chart for the payment methods: Question 3
st.subheader('The most popular payment method')
values= data['Payment'].value_counts().tolist()
labels= data['Payment'].unique().tolist()

fig= {
    'data': [
        {
            "values": values, 
            "labels": labels, 
            "domain": {"x":[0,.5]},
            "name": "Payment", 
            "hoverinfo": "label+percent+name", 
            "type":"pie"
           
        },],
      "layout": {
   
      }
}
st.plotly_chart(fig)

#To construct the boxplot of each branch in terms of gross income: Question 4
st.subheader('The most profitable branch')
ax= sns.boxplot(x=data['Branch'], y=data['gross income'], palette="Set2", width=0.5) 
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
st.pyplot()

#To construct the boxplot of each product line in terms of gross income: Question 5 
st.subheader('The most profitable product line')
ax= sns.boxplot(x=data['Product line'],y= data['gross income'], data=pd.melt(data), palette="Set2",width=0.5)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=6.5)
plt.tight_layout()
st.pyplot()

    
#To know the maximum number of products bought at once: Question 6
max_quantity= data['Quantity'].nlargest(n=1)

#To lay out widgets side-by-side:
st.subheader("The maximum number of products bought at once")
left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Answer')
if pressed:

    st.write('10 products')


#To specify the top 3 product lines: Question 8
top_3= data['Product line'].value_counts()[:3]

st.subheader("Top product lines")
left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Know more')
if pressed:

    st.write('Fashion & Accessories,','Food & Beverages,','and Electronic Accessories')

#To know which product line has the highest ratings: Question 17 
st.subheader("Product line with the highest ratings")
left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Show answer')
if pressed:

    st.write('Food and beverage')
 
       
#To construct a bar chart showing the customer type: Question 7
st.subheader("Customer type distribution")
values= data['Customer type'].value_counts().tolist()
labels= data['Customer type'].unique().tolist()

fig= {
    'data': [
        {
            "values": values, 
            "labels": labels, 
            "domain": {"x":[0,.5]},
            "name": "Customer Type", 
            "hoverinfo": "label+percent+name", 
            "type":"pie"
        },],
      "layout": {
   
      }
}
st.plotly_chart(fig)

#To know if females more likely to have membership cards than males: Question 11
st.subheader("Membership across genders")
female_members= data.loc[(data['Customer type']== 'Member') & (data['Gender']== 'Female')]
female_label = female_members['Gender'].unique()
female_counts = female_members['Gender'].value_counts().tolist()

male_members= data.loc[(data['Customer type']== 'Member') & (data['Gender']== 'Male')]
male_label = male_members['Gender'].unique()
male_counts = male_members['Gender'].value_counts().tolist()

fig = make_subplots(rows = 1, 
                    cols = 2, 
                    column_widths = [4, 4], 
                    shared_yaxes = True, 
                    specs=[[{"secondary_y": True}, {"secondary_y": True}]]
                   
                  
)

#Females

fig.add_trace(go.Bar(
    x = female_label,
    y = female_counts,
    name = 'Female Membership',
    marker_color = '#1c9099',
), row = 1, col = 1, secondary_y = False)

#Males
fig.add_trace(go.Bar(
    x = male_label,
    y = male_counts,
    name = 'Male Membership',
    marker_color = '#a8ddb5',#green_shades,
), row = 1, col = 2, secondary_y = False)

st.plotly_chart(fig)


#To construct a barchart showing the gender: Question 9 
st.subheader("Gender distribution")
gen=['Female','Male']
fig = go.Figure([go.Bar(x= gen, y=[501, 499],marker_color="#756bb1", width=0.5)])
st.plotly_chart(fig)


#To know if males and females exhibit different shopping patterns: Question 10
st.subheader("Shopping patterns across genders")
#Females:
female_products= data.loc[data['Gender']== 'Female']
female_productline = female_products['Product line'].unique()
female_productline_counts = female_products['Product line'].value_counts().tolist()

#Males:
male_products= data.loc[data['Gender']== 'Male']
male_productline = male_products['Product line'].unique()
male_productline_counts = male_products['Product line'].value_counts().tolist()

#To plot the results:
fig = make_subplots(rows = 1, 
                    cols = 2, 
                    column_widths = [4, 4], 
                    shared_yaxes = True, 
                    specs=[[{"secondary_y": True}, {"secondary_y": True}]]
                  
)

#Females
fig.add_trace(go.Bar(
    x = female_productline,
    y = female_productline_counts,
    name = 'Female Purchases',
    marker_color = '#1c9099',
    text = ['🎖', '🎖', '🎖'],
), row = 1, col = 1, secondary_y = False)

#Males:
fig.add_trace(go.Bar(
    x = male_productline,
    y = male_productline_counts,
    name = 'Male Purchases',
    marker_color = '#a8ddb5',#green_shades,
    text = ['🎖', '🎖'],
), row = 1, col = 2, secondary_y = False)

st.plotly_chart(fig)


#To know if there is a correlation between the variables: Question 14
st.subheader("Correlation analysis")
sns.heatmap(np.round(data.corr(),2), annot=True)
st.pyplot()
st.write("The unit price is positively correlated to COGS with a 63% correlation. Also, quantity is a 71% correlation with the gross income. However, rating doesn\'t have any correlation with any other variable.")

#To know if sales  are affected by the days the week: Question 15
st.subheader("Sales across days")
data['Date'] = pd.to_datetime(data['Date'])
data['weekday'] = data['Date'].dt.day_name()
plt.figure(figsize=(9, 7))
plt.title('Total Sales by Day of the Week')
sns.countplot(data['weekday'],palette="Set2")
st.pyplot()

#To know which is the busiest time of the day: Question 16
st.subheader("The busiest time of the day")             
data['Time'] = pd.to_datetime(data['Time'])
data['Hour'] = (data['Time']).dt.hour
data['Hour'].unique()
sns.lineplot(x="Hour", y = 'Quantity',data=data).set_title("Total Sales per Hour")
st.pyplot()

st.text("")
st.subheader('**Growth Options**')
st.text("")
#To know which city has the most potential in terms of gross income: Question 12
st.subheader("Most promising city to expand into")
fig = go.Figure(go.Bar(x=[5265.18, 5057.16, 5057.03] , y=['Naypyitaw','Yangon','Mandalay'],marker_color='#fdbb84', width=0.5, orientation='h'))
st.plotly_chart(fig)

#Which product line should they focus on in this city: Question 13
st.subheader("Product lines analysis")
if st.checkbox('Most Promising Line'):
    st.write('The most promising branch is food and beverage with 1,132 $ per month.')
if st.checkbox('Least Promising Line'):
    st.write('The least promising branch is Home and lifestyle with 661.7 $ per month.')



#RFM Analysis:

#To read the dataset:
orig_url= 'https://drive.google.com/file/d/1rUCLv0JsfJWcM2hWd56vV9Ul89R8kigF/view?usp=sharing'

file_id = orig_url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url = requests.get(dwn_url).text
csv_raw = StringIO(url)
data= pd.read_csv(csv_raw)

data['Date'] = pd.to_datetime(data['Date'])
max(data['Date'])
#To add 1 day to the max date:
pin_date = max(data['Date']) + dt.timedelta(1)

#To create the RFM dataframe:
rfm = data.groupby('Invoice ID').agg({
    'Date': lambda x: (pin_date - x.max()).days,
    'Invoice ID': 'count',
    'cogs': 'sum'
})

#To rename the columns:
rfm.rename(columns={'Invoice ID': 'Frequency', 'cogs': 'Monetary', 'Date': 'Recency' }, inplace=True)

#To create a new column:
rfm["RecencyScore"] = pd.qcut(rfm['Recency'],5, labels = [5,4,3,2,1])
#To create a new column:
cut_bins = [0,1,2,3,9,10]
rfm["FrequencyScore"] = pd.cut(rfm["Frequency"],bins = cut_bins, labels = [1, 2, 3, 4, 5])
#To create a new column:
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
rfm.head()

#To combine the scores:
rfm["RFM Score"] = (rfm["RecencyScore"].astype(str) +
                    rfm["FrequencyScore"].astype(str) +
                    rfm["MonetaryScore"].astype(str))


#To know the count num of unique segments:
rfm_count_unique = rfm.groupby('RFM Score')['RFM Score'].nunique()
print(rfm_count_unique.sum())

#To calculate RFM Score:
rfm['RFM Score'] = rfm[['RecencyScore','FrequencyScore','MonetaryScore']].sum(axis=1)
print(rfm['RFM Score'].head())

#To define rfm level function:
def rfm_level(df):
    if df['RFM Score'] >= 9:
        return 'Champions'
    elif ((df['RFM Score'] >= 8) and (df['RFM Score'] <= 9)):
        return 'Can\'t lose them'
    elif ((df['RFM Score'] >= 7) and (df['RFM Score'] < 8)):
        return 'Loyal'
    elif ((df['RFM Score'] >= 6) and (df['RFM Score'] < 7)):
        return 'Potential'
    elif ((df['RFM Score'] >= 5) and (df['RFM Score'] < 6)):
        return 'Promising'
    elif ((df['RFM Score'] >= 4) and (df['RFM Score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Require Activation'
    
#To create a new variable RFM Level
rfm['RFM Level'] = rfm.apply(rfm_level, axis=1)

#Calculate average values for each RFM_Level, and return a size of each segment: 
rfm_level_agg = rfm.groupby('RFM Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(1)
# Print the aggregated dataset:
print(rfm_level_agg)

#To plot a heatmap: 
st.subheader("**Customer Segments**")
st.text("")
st.write("These customer segments are the result of the RFM analysis which aims to segment customers based on ""Frequency"", ""Recency"", and ""Monetary value"". It numerically ranks a customer from 1-5 with 5 being best score to better target customers.")
rfm_level_agg.columns = rfm_level_agg.columns
rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=rfm_level_agg['Count'], 
              label=['Champions',
                     'Can\'t Lose Them',
                     'Loyal',
                     'Needs Attention',
                     'Potential', 
                     'Promising', 
                     'Require Activation'], alpha=.6 )
plt.title("RFM Segments",fontsize=18,fontweight="bold")
plt.axis('off')
st.pyplot()

#Suitable Strategies for each segment: 
st.subheader('Suitable strategy for each segment')
product= rfm["RFM Level"].unique()
option = st.selectbox(
    'Which segment do you want to choose?',
     product)

if "Can\'t lose them" in option:
    st.write('They are valuable customers with average spending and average recency. Try making offers on the items that they like to encourage them to spend more and become Champions!')
if "Champions" in option:
    st.write('They are the most valuable customers who bought most recently and are heavy spenders. Try rewarding them to promote our stores!')
if "Loyal" in option:
    st.write('They are average spenders, but are not frequent shoppers. Try offering membership programs and support to encourage them to visit the stores more often!')
if "Needs Attention" in option:
    st.write('They made their first purchase but have not come back agian. Try reaching out again to create some brand awareness with them!')
if "Potential" in option:
    st.write('They have a high potential to enter our loyal segment, try giving them some freebie items on their next purchase to show them that they are valuable!')
if "Promising" in option:
    st.write('They promising signs with quantity and value of their purchases, but they haven\'t bought for a long time. Try targeting them with their wishlist items or limited time offers!')
if "Require Activation" in option:
    st.write('They have the lowest RFM score. Maybe they switch to our competitors. Try implementing a different strategy to win them back!')
        


#Level Based Persona: 

st.subheader("**Level Based Persona**")
st.text("")
st.write("This a simple customer segmentation strategy which aims to segment customers based on their potential revenue generation. The characteristics of each unique persona are gathered with the expected average revenue resulting in the segment score. The segments are from 1-3 with 1 being the most profitable.")

#To create a dataframe that has only categorical features:
categorical= data[['Branch','Customer type','Gender','Product line','Payment','Rating']]

#To add the average total sales and sort the dataframe by that column:
cat_total = data.groupby(by=['Branch', 'Customer type',
                          'Gender', 'Product line', 'Payment', 'Rating']).\
                          agg({"Total" : "sum"}).sort_values("Total", ascending=False)
cat_total.reset_index(inplace=True)

#To convert the Rating to categorical:
bins = [cat_total["Rating"].min(), 5.5, 7, 8.5, cat_total["Rating"].max()]
labels = [str(cat_total["Rating"].min()) + '-5.5',
            '5.5-7', '7-8.5',
            '8.5-'+str(cat_total["Rating"].max())]
cat_total["Rating Range"] = pd.cut(cat_total["Rating"], bins, labels=labels)

#To create a persona column of these characteristics: 
cat_total['Persona'] = [row[0] + "_" + row[1] + "_"
                                  + row[2] + "_" + row[3] + "_"
                                  + row[4]+ "_" 
                                  + str(row[7]) for row in cat_total.values]

#To create a dataframe with Persona and total:
cat_total = cat_total.reset_index()
cat_total = cat_total[["Persona", "Total"]]

#To find the average total revenues according to the Persona segmentation:
cat_total = cat_total.groupby("Persona").agg({"Total": "mean"})
cat_total.head(10)

#To add the Segments:
cat_total= cat_total.reset_index()
cat_total["Segment"] = pd.qcut(cat_total["Total"], 4, labels = ["4", "3", "2", "1"])

product= cat_total["Persona"].unique()
option = st.selectbox(
    'Which possible persona do you want to choose?',
     product)

if cat_total['Persona'].iloc[0] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[0],"and the average expected revenue is", cat_total['Total'].iloc[0],"$")
    
if cat_total['Persona'].iloc[1] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[1],"and the average expected revenue is", cat_total['Total'].iloc[1],"$")
    
if cat_total['Persona'].iloc[2] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[2],"and the average expected revenue is", cat_total['Total'].iloc[2],"$")
    
if cat_total['Persona'].iloc[3] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[3],"and the average expected revenue is", cat_total['Total'].iloc[3],"$")
    
if cat_total['Persona'].iloc[4] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[4],"and the average expected revenue is", cat_total['Total'].iloc[4],"$")
    
if cat_total['Persona'].iloc[5] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[5],"and the average expected revenue is", cat_total['Total'].iloc[5],"$")
    
if cat_total['Persona'].iloc[6] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[6],"and the average expected revenue is", cat_total['Total'].iloc[6],"$")
    
if cat_total['Persona'].iloc[7] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[7],"and the average expected revenue is", cat_total['Total'].iloc[7],"$")
if cat_total['Persona'].iloc[8] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[8],"and the average expected revenue is", cat_total['Total'].iloc[8],"$")
    
if cat_total['Persona'].iloc[9] in option:
    st.write("The Segment is", cat_total['Segment'].iloc[9],"and the average expected revenue is", cat_total['Total'].iloc[9],"$")


#To try a possible persona:
new_user = "B_Normal_Female_Sports and travel_Ewallet_5.5-7"
segment= cat_total[cat_total["Persona"] == new_user]

    