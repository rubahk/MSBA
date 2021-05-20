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
st.write("by Ruba Al Hakeem| May 2021")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("")
st.markdown("A large supermarket in Myanmar has three branches in Yangon, Naypyitaw, and  Mandalay. Its managers want to solve a Marketing and Sales problem related to customer segmentation and marketing strategies.")
  


#To create the sidebar menu:
st.sidebar.title("Menu")
pressed= st.sidebar.radio("Navigate",["Data Visualization", "Customer Segmentation"])
st.sidebar.title("About")
st.sidebar.info(
    """
    This dashboard aims to visualize and analyze different business questions related to profitablity and customer segmentation of a large supermarket in Myanmar. 
    """
)


#To know the total sales per branch:  
Branch= data['Branch'].unique()
sale= data.groupby('Branch')['Total'].sum().round()
A=sale.iloc[0]
B=sale.iloc[1]
C=sale.iloc[2]

#To know the total sales per branches: 
col1, col2 = st.beta_columns([0.5,0.5])
if "Data Visualization" in pressed:
    with col1:
        st.subheader('Total sales per branch')
        opt = st.selectbox('Which branch do you want to choose?',Branch)
        if "A" in opt:
            st.write('The total sales in dollars of this branch is', A)
        if "B" in opt:
            st.write('The total sales in dollars of this branch is', B)
        if "C" in opt:
            st.write('The total sales in dollars of this branch is', C)
   
    

#To know the total sales per product line: 
branch=data.groupby('Product line')['Total'].sum().round()
E=branch.iloc[0]
Fa=branch.iloc[1]
Fo=branch.iloc[2]
He=branch.iloc[3]
Ho=branch.iloc[4]
S=branch.iloc[5]
product= data['Product line'].unique()
if "Data Visualization" in pressed:
    
    with col2:
        st.subheader('Total sales per product line')
        option = st.selectbox('Which product line do you want to choose?', product)
        
        if "Health and beauty" in option:
            st.write('The total sales in dollars of this product line is', He)
        if "Electronic accessories" in option:
            st.write('The total sales in dollars of this product line is', E)
        if "Home and lifestyle" in option:
            st.write('The total sales in dollars of this product line is', Ho)
        if "Sports and travel" in option:
            st.write('The total sales in dollars of this product line is', S)
        if "Food and beverages" in option:
            st.write('The total sales in dollars of this product line is',Fo)
        if "Fashion accessories" in option:
            st.write('The total sales in dollars of this product line is',Fa)
  
    
st.markdown('#')

#To create the filter: 
st.sidebar.title("Select one branch to explore it!")
neighborhood = st.sidebar.radio("Branch", data['Branch'].unique())


#To know the profitability of per branch in terms of gross income:   
branch=data.groupby('Branch')['gross income'].sum()
branch=branch.reset_index()


if "Data Visualization" in pressed:
    col3, col4 = st.beta_columns([0.5,0.5])
    with col3:
        if 'A' in neighborhood:
                st.subheader('Profitability per branch')
                fig= px.bar(branch, x=branch["Branch"],y=branch["gross income"],width=400,height=400,labels={"y":"Gross Income ($)","x":"Branch"})
                fig.update_traces(marker_color='#67a9cf')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)", 
                })
                st.plotly_chart(fig)
                
                
               
        if 'B' in neighborhood:
                st.subheader('Profitability per branch')
                fig= px.bar(branch, x=data["Branch"],y=data["gross income"],width=400,height=400,labels={"y":"Gross Income ($)","x":"Branch"})
                fig.update_traces(marker_color='#67a9cf')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)", 
                })
                st.plotly_chart(fig)
                
        if 'C' in neighborhood:
                st.subheader('Profitability per branch')
                fig= px.bar(branch, x=data["Branch"],y=data["gross income"],width=400,height=400,labels={"y":"Gross Income ($)","x":"Branch"})
                fig.update_traces(marker_color='#67a9cf')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)", 
                })
                st.plotly_chart(fig)
                   
            
              
        
#To know the profitability each product line in terms of gross income: 
A_lines=data[data['Branch']=='A']
A_lines=A_lines.groupby('Product line')['gross income'].sum()
A_lines=A_lines.reset_index() 

B_lines=data[data['Branch']=='B']
B_lines=B_lines.groupby('Product line')['gross income'].sum()
B_lines=B_lines.reset_index() 

C_lines=data[data['Branch']=='C']
C_lines=C_lines.groupby('Product line')['gross income'].sum()
C_lines=C_lines.reset_index() 

if "Data Visualization" in pressed:
    with col4:
        if 'A' in neighborhood:
           
                st.subheader('Profitability per product line')
                fig= px.bar(A_lines, x=A_lines["Product line"],y=A_lines["gross income"],width=400,height=400,labels={"y":"Gross Income($)","x":"Product Line"})
                fig.update_traces(marker_color='#ef8a62')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)",
                })
                st.plotly_chart(fig)
                
        if 'B' in neighborhood:
            
                st.subheader('Profitability per product line')
                fig= px.bar(B_lines, x=B_lines["Product line"],y=B_lines["gross income"],width=400,height=400,labels={"y":"Gross Income($)","x":"Product Line"})
                fig.update_traces(marker_color='#ef8a62')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)", 
                })
                st.plotly_chart(fig)
                
                
        if 'C' in neighborhood:
            
                st.subheader('Profitability per product line')
                fig= px.bar(C_lines, x=C_lines["Product line"],y=C_lines["gross income"],width=400,height=400,labels={"y":"Gross Income($)","x":"Product Line"})
                fig.update_traces(marker_color='#ef8a62')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)", 
                })
                st.plotly_chart(fig)
  
    
#To construct a bar chart showing the customer type: 
A_cust=data[data['Branch']=='A']
valuesA= A_cust['Customer type'].value_counts().tolist()
labelsA= A_cust['Customer type'].unique().tolist()

B_cust=data[data['Branch']=='B']
valuesB= B_cust['Customer type'].value_counts().tolist()
labelsB= B_cust['Customer type'].unique().tolist()

C_cust=data[data['Branch']=='C']
valuesC= C_cust['Customer type'].value_counts().tolist()
labelsC= C_cust['Customer type'].unique().tolist()

if "Data Visualization" in pressed:
    col5, col6 = st.beta_columns([0.5,0.5])  
    
    with col5:
        
        if 'A' in neighborhood:
            st.subheader("Customer type distribution")
            fig= px.bar(A_cust, x=labelsA,y=valuesA,width=400,height=400,labels={"y":"Number of customers","x":"Customer Type"})
            fig.update_traces(marker_color='#5ab4ac')
            fig.update_layout({
            "plot_bgcolor":"rgba(0, 0, 0, 0)",
            "paper_bgcolor":"rgba(0, 0, 0, 0)", 
            })
            st.plotly_chart(fig)
            
     
        
        if 'B' in neighborhood:
            st.subheader("Customer type distribution")
            fig= px.bar(B_cust, x=labelsB,y=valuesB,width=400,height=400,labels={"y":"Number of customers","x":"Customer Type"})
            fig.update_traces(marker_color='#5ab4ac')
            fig.update_layout({
            "plot_bgcolor":"rgba(0, 0, 0, 0)",
            "paper_bgcolor":"rgba(0, 0, 0, 0)", 
            })
            st.plotly_chart(fig)
   
      
        if 'C' in neighborhood:
            st.subheader("Customer type distribution")
            fig= px.bar(C_cust, x=labelsC,y=valuesC,width=400,height=400,labels={"y":"Number of customers","x":"Customer Type"})
            fig.update_traces(marker_color='#5ab4ac')
            fig.update_layout({
            "plot_bgcolor":"rgba(0, 0, 0, 0)",
            "paper_bgcolor":"rgba(0, 0, 0, 0)", 
            })
            st.plotly_chart(fig)
                
        
#To know if females more likely to have membership cards than males: 
membersA=data[data['Branch']=='A']
membersA=membersA.loc[(membersA['Customer type']== 'Member')]
membersA=membersA.groupby('Gender')['Customer type'].count()
membersA=membersA.reset_index()                   

membersB=data[data['Branch']=='B']
membersB=membersB.loc[(membersB['Customer type']== 'Member')]
membersB=membersB.groupby('Gender')['Customer type'].count()
membersB=membersB.reset_index() 

membersC=data[data['Branch']=='C']
membersC=membersC.loc[(membersC['Customer type']== 'Member')]
membersC=membersC.groupby('Gender')['Customer type'].count()
membersC=membersC.reset_index() 


if "Data Visualization" in pressed:
     with col6:
            if 'A' in neighborhood:
                st.subheader("Membership across genders")
                fig= px.bar(membersA, x=membersA['Gender'],y=membersA['Customer type'],width=400,height=400)
                fig.update_traces(marker_color='#d8b365')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)", 
                })
                st.plotly_chart(fig)
                
            if 'B' in neighborhood:
                st.subheader("Membership across genders")
                fig= px.bar(membersB, x=membersB['Gender'],y=membersB['Customer type'],width=400,height=400)
                fig.update_traces(marker_color='#d8b365')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)", 
                })
                st.plotly_chart(fig)
                
            if 'C' in neighborhood:
                st.subheader("Membership across genders")
                fig= px.bar(membersC, x=membersC['Gender'],y=membersC['Customer type'],width=400,height=400)
                fig.update_traces(marker_color='#d8b365')
                fig.update_layout({
                "plot_bgcolor":"rgba(0, 0, 0, 0)",
                "paper_bgcolor":"rgba(0, 0, 0, 0)", 
                })
                st.plotly_chart(fig)
            
   
    

#To construct a barchart showing the gender: 
Gen_A=data[data['Branch']=='A']
valuesA= Gen_A['Gender'].value_counts().tolist()
labelsA= Gen_A['Gender'].unique().tolist()

Gen_B=data[data['Branch']=='B']
valuesB= Gen_B['Gender'].value_counts().tolist()
labelsB=Gen_B['Gender'].unique().tolist()

Gen_C=data[data['Branch']=='C']
valuesC= Gen_C['Gender'].value_counts().tolist()
labelsC= Gen_C['Gender'].unique().tolist()

if "Data Visualization" in pressed: 
    col7, col8 = st.beta_columns([0.5,0.5]) 
    with col7:
        if 'A' in neighborhood:
            st.subheader('Gender distribution')
            fig= px.bar(Gen_A, x=labelsA,y=valuesA,width=400,height=400,labels={"y":"Number of customers","x":"Gender"})
            fig.update_traces(marker_color='#af8dc3')
            fig.update_layout({
            "plot_bgcolor":"rgba(0, 0, 0, 0)",
            "paper_bgcolor":"rgba(0, 0, 0, 0)",
            })
            st.plotly_chart(fig)
            
        if 'B' in neighborhood:
            st.subheader('Gender distribution')
            fig= px.bar(Gen_B, x=labelsB,y=valuesB,width=400,height=400,labels={"y":"Number of customers","x":"Gender"})
            fig.update_traces(marker_color='#af8dc3')
            fig.update_layout({
            "plot_bgcolor":"rgba(0, 0, 0, 0)",
            "paper_bgcolor":"rgba(0, 0, 0, 0)",
            })
            st.plotly_chart(fig)
            
        if 'C' in neighborhood:
            st.subheader('Gender distribution')
            fig= px.bar(Gen_C, x=labelsC,y=valuesC,width=400,height=400,labels={"y":"Number of customers","x":"Gender"})
            fig.update_traces(marker_color='#af8dc3')
            fig.update_layout({
            "plot_bgcolor":"rgba(0, 0, 0, 0)",
            "paper_bgcolor":"rgba(0, 0, 0, 0)",
            })
            st.plotly_chart(fig)
            
    
          
    
#To construct a pie chart for the payment methods: 
PA=data[data['Branch']=='A']
PB=data[data['Branch']=='B']
PC=data[data['Branch']=='C']

if "Data Visualization" in pressed: 
    with col8:
        if 'A' in neighborhood:
            st.subheader('Payment methods')
            values= PA['Payment'].value_counts().tolist()
            labels= PA['Payment'].unique().tolist()

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

        if 'B' in neighborhood:
            st.subheader('Payment methods')
            values= PB['Payment'].value_counts().tolist()
            labels= PB['Payment'].unique().tolist()

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
            
        if 'C' in neighborhood:
            st.subheader('Payment method')
            values= PC['Payment'].value_counts().tolist()
            labels= PC['Payment'].unique().tolist()

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
            
  
 
#To know if sales  are affected by the days the week: 
StoreA=data[data['Branch']=='A']
StoreA['Date'] = pd.to_datetime(StoreA['Date'])
StoreA['weekday']= StoreA['Date'].dt.day_name().tolist()

StoreB=data[data['Branch']=='B']
StoreB['Date'] = pd.to_datetime(StoreB['Date'])
StoreB['weekday']= StoreB['Date'].dt.day_name().tolist()

StoreC=data[data['Branch']=='C']
StoreC['Date'] = pd.to_datetime(StoreC['Date'])
StoreC['weekday']= StoreC['Date'].dt.day_name().tolist()

if "Data Visualization" in pressed:  
    col9, col10 = st.beta_columns([0.5,0.5])
    with col9:
        if 'A' in neighborhood:
            st.subheader("Total sales across days")
            fig= px.bar(StoreA, x=StoreA['weekday'].unique().tolist(),y=StoreA.groupby('weekday')['Total'].sum().tolist(),width=400,height=400,labels={"y":"Sales","x":"Day"})
            fig.update_traces(marker_color='#e9a3c9')
            fig.update_layout({
           "plot_bgcolor":"rgba(0, 0, 0, 0)",
           "paper_bgcolor":"rgba(0, 0, 0, 0)", 
           })
            st.plotly_chart(fig)
            
        if 'B' in neighborhood:
            st.subheader("Total sales across days")
            fig= px.bar(StoreB, x=StoreB['weekday'].unique().tolist(),y=StoreB.groupby('weekday')['Total'].sum().tolist(),width=400,height=400,labels={"y":"Sales","x":"Day"})
            fig.update_traces(marker_color='#e9a3c9')
            fig.update_layout({
           "plot_bgcolor":"rgba(0, 0, 0, 0)",
           "paper_bgcolor":"rgba(0, 0, 0, 0)", 
           })
            st.plotly_chart(fig)
            
            
        if 'C' in neighborhood:
            st.subheader("Total sales across days")
            fig= px.bar(StoreC, x=StoreC['weekday'].unique().tolist(),y=StoreC.groupby('weekday')['Total'].sum().tolist(),width=400,height=400,labels={"y":"Sales","x":"Day"})
            fig.update_traces(marker_color='#e9a3c9')
            fig.update_layout({
           "plot_bgcolor":"rgba(0, 0, 0, 0)",
           "paper_bgcolor":"rgba(0, 0, 0, 0)", 
           })
            st.plotly_chart(fig)
        

#To know which is the busiest time of the day: 
SA=data[data['Branch']=='A']
SA['Time'] = pd.to_datetime(SA['Time'])
SA['Hour'] = (SA['Time']).dt.hour.tolist()
countA=SA.groupby('Hour')['Invoice ID']

SB=data[data['Branch']=='B']
SB['Time'] = pd.to_datetime(SB['Time'])
SB['Hour'] = (SB['Time']).dt.hour.tolist()
countB=SB.groupby('Hour')['Invoice ID']

SC=data[data['Branch']=='C']
SC['Time'] = pd.to_datetime(SC['Time'])
SC['Hour'] = (SC['Time']).dt.hour.tolist()
countC=SC.groupby('Hour')['Invoice ID']

if "Data Visualization" in pressed: 
    with col10:
        if 'A' in neighborhood:
            st.subheader("The busiest time of the day")             
            fig= px.scatter(SA,x=SA["Hour"].unique().tolist(), y=SA.groupby('Hour')['Invoice ID'].count().tolist(),labels={"y":"Number of customers","x":"Hour"})
            fig.update_traces(marker_color='#67a9cf')
            st.plotly_chart(fig) 
        if 'B' in neighborhood:
            st.subheader("The busiest time of the day")             
            fig=px.scatter(SB,x=SB["Hour"].unique().tolist(), y=SB.groupby('Hour')['Invoice ID'].count().tolist(), labels={"y":"Number of customers","x":"Hour"})
            fig.update_traces(marker_color='#67a9cf')
            st.plotly_chart(fig)
            
        if 'C' in neighborhood:
            st.subheader("The busiest time of the day")             
            fig=px.scatter(SC,x=SC["Hour"].unique().tolist(), y=SC.groupby('Hour')['Invoice ID'].count().tolist(),labels={"y":"Number of  customers","x":"Hour"})
            fig.update_traces(marker_color='#67a9cf')
            st.plotly_chart(fig)


    
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

#To create a new column for the recency score: 
#The label of the recency scores must be inversed--the more recent the better

# Create numbered labels
r_labels = list(range(4, 0, -1))
# Divide into groups based on quartiles
recency_quartiles = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
# Create new column
rfm['RecencyScore'] = recency_quartiles
# Sort recency values from lowest to highest
rfm.sort_values('Recency')

#To create a new column for the frequency score using the q=4 to get the quartiles:
#Because the frequency is 1 in this case, I will directly assign 1 to frequencyScore the qcut is not working

# Create new column for frequency:
rfm['FrequencyScore'] = rfm['Frequency']
# Sort recency values from lowest to highest
rfm.sort_values('Frequency')

#To create a new column for the monetary score  using the q=4 to get the quartiles:
m_labels = range(1,5)
monetary_quartiles = pd.qcut(rfm['Monetary'], 4, labels = m_labels)
#To create a new column for monetary:
rfm['MonetaryScore'] = monetary_quartiles
rfm.sort_values('Monetary')

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
if "Customer Segmentation" in pressed:
    st.subheader("Customer Segments")
    st.write("These customer segments are the result of the RFM analysis which aims to segment customers based on ""Frequency"", ""Recency"", and ""Monetary value"". It numerically ranks a customer from 1-4 with 4 being best score to better target customers.")
    rfm_level_agg.columns = rfm_level_agg.columns
    rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
#Create our plot and resize it.
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(16, 9)
    squarify.plot(sizes=rfm_level_agg['Count'], 
              label=['Can\'t Lose Them',
                     'Champions',
                     'Loyal',
                     'Needs Attention',
                     'Potential', 
                     'Promising', 
                     'Require Activation'], alpha=.6 )
    plt.title("RFM Segments",fontsize=18,fontweight="bold")
    plt.axis('off')
    st.pyplot()


#Suitable Strategies for each segment:
if "Customer Segmentation" in pressed:
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