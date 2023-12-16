import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")
st.header("WORKMIND METRICS: LOGISTIC REGRESSION")

st.sidebar.image("images/logo2.png",caption="EmployeeEcho Insights")

theme_plotly = None 

# load Style css
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


df = pd.read_excel('random_data.xlsx')

# Encoding categorical variables
label_encoder = LabelEncoder()
df['X1'] = label_encoder.fit_transform(df['X1'])
df['X2'] = label_encoder.fit_transform(df['X2'])
df['Y'] = label_encoder.fit_transform(df['Y'])

# Logistic Regression
X = df[['X1', 'X2']]
Y = df['Y']

log_reg = LogisticRegression()
log_reg.fit(X, Y)

# Prediction and Residual Calculation
predicted_Y = log_reg.predict(X)
residuals = Y - predicted_Y

# Calculating Sum of Squares
SSE = mean_squared_error(Y, predicted_Y) * len(Y)
SSR = np.sum((predicted_Y - np.mean(Y)) ** 2)
SST = SSE + SSR

# Coefficients and Statistics
coefficients = log_reg.coef_[0]
intercept = log_reg.intercept_

# Calculating R-squared
r_squared = r2_score(Y, predicted_Y)

# Adjusted R-squared
n = len(Y)

p = 2  # Number of predictors (X1 and X2)
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Coefficient of Determination
coefficient_determination = 1 - SSE / SST

# Correlation Coefficient
correlation_coefficient = np.sqrt(coefficient_determination)

# Predicting Y values
predicted_Y = log_reg.predict(X)

# Adding predicted Y to the DataFrame
df['Predicted_Y'] = predicted_Y

# Output Table

df2 = pd.read_excel('random_data.xlsx')
output_table = pd.DataFrame({
    'Promotions': df2['X1'],
    'Salary Increment': df2['X2'],
    'Job satisfaction': df2['Y'],
    'Job satisfaction2': df['Y'],
    'Predicted satisfaction':df['Predicted_Y'],
    'Residual': residuals
})

#print regression coefficients
st.success(f"**Regression Coefficients** (B1x1, B2x2): {coefficients}")

#show predicted table
with st.expander("LOGISTIC PREDICTION TABLE"):
 showData=st.multiselect('Filter: ',output_table.columns,default=["Promotions","Salary Increment","Job satisfaction","Job satisfaction2","Predicted satisfaction","Residual"])
 st.dataframe(output_table[showData],use_container_width=True)
 
#show metrics
c1,c2,c3,c4=st.columns(4)
c1.metric(f"R-squared:",value= f"{r_squared:,.3f}")
c2.metric(f"Adjusted R-squared:", value=f"{adjusted_r_squared:,.3f}")
c3.metric(f"Coefficient of Determination:",value= f"{coefficient_determination*100:,.1f} %")
c4.metric(f"Correlation Coefficient:", value=f"{correlation_coefficient:,.3f}")
style_metric_cards()

#check correlation
if correlation_coefficient > 0.8:
    st.success("Strong positive Correlation")
elif correlation_coefficient > 0.5:
    st.success("Moderate positive Correlation")
elif correlation_coefficient > 0:
    st.success("Weak positive Correlation")
elif correlation_coefficient == 0:
    st.success("No relationship Correlation")
elif correlation_coefficient < -0.8:
    st.success("Strong negative Correlation")
elif correlation_coefficient < -0.5:
    st.success("Moderate negative Correlation")
elif correlation_coefficient < 0:
    st.success("Weak negative Correlation")

#create plots
import plotly.express as px
grouped = df2.groupby(['Y', 'X2', 'X1']).size().reset_index(name='count')
fig = px.bar(grouped, x='Y', y='count', color='X1', barmode='stack')
fig.update_layout(
    title='Promotions',
    xaxis_title='X2',
    yaxis_title='Count',
    legend_title='X1'
)

fig2 = px.bar(grouped, x='Y', y='count', color='X2', barmode='stack')
fig2.update_layout(
    title='Salary Increments',
    xaxis_title='X2',
    yaxis_title='Count',
    legend_title='X1'
)

b1,b2=st.columns(2)
b1.plotly_chart(fig,use_container_width=True)
b2.plotly_chart(fig2,use_container_width=True)