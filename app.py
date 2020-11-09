# -*- coding: utf-8 -*-
"""
@author: raghav
"""

from flask import Flask, render_template, request
import numpy as np
import pickle 

model = pickle.load(open('loanPrediction.pkl','rb'))
 
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        pass
        # Failure to return a redirect or render_template
    else:
        return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        #Gender
        Gender_Male = request.form['Gender']
        #print("\n",Gender_Male)
        if Gender_Male == 'Male':
            Gender_Male = 1
        elif Gender_Male == 'Female':
            Gender_Male = 0

        Gender_Male = np.uint8(Gender_Male)

        # Education
        Education = request.form['Education']
        if Education == "Graduate":
            Education = 0
        else:
            Education = 1

        Education = np.uint8(Education)
        
        #Married
        Married = request.form['Married']
        if Married == "Yes":
            Married = 1
        else:
            Married = 0
        Married = np.uint8(Married)

        # self employed
        Self_Employed = request.form['SelfEmployed']
        if Self_Employed == "Yes":
            Self_Employed = 1
        else:
            Self_Employed = 0

        Self_Employed = np.uint8(Self_Employed)

        #Dependents
        Dependents = request.form['Dependents']
        Dependents = np.int32(Dependents)

        #Property Area
        Property_Area = request.form['PropertyArea']
        if Property_Area == "Urban":
            Property_Area_Semiurban = 0
            Property_Area_Urban = 1
        elif Property_Area == "SemiUrban":
            Property_Area_Semiurban = 1
            Property_Area_Urban = 0
        else:
            Property_Area_Semiurban = 0
            Property_Area_Urban = 0
        
        Property_Area_Semiurban = np.uint8(Property_Area_Semiurban)
        Property_Area_Urban = np.uint8(Property_Area_Urban)

        #credit History
        Credit_History = request.form['CreditHistory']
        if Credit_History == "NotPaid":
            Credit_History = 0
        elif Credit_History == "Paid":
            Credit_History = 1
        else:
            Credit_History = 1
        Credit_History = np.int32(Credit_History)

        #Loan Amount
        Loan_Amount = request.form['LoanAmount']
        Loan_Amount = np.float64(Loan_Amount)
        Loan_Amount = Loan_Amount  / 1000
        loan_log = np.log(Loan_Amount)
        loan_log = np.float64(loan_log)

        #Applicant Income
        Applicant_Income = request.form['ApplicantIncome']
        Applicant_Income = np.float64(Applicant_Income)
        Applicant_Income = Applicant_Income / 10
        if Applicant_Income <= 2500:
            income_bin_average = 0
            income_bin_high = 0
            income_bin_very_high = 0
        elif Applicant_Income <= 5000 and Applicant_Income > 2500:
            income_bin_average = 1
            income_bin_high = 0
            income_bin_very_high = 0
        elif Applicant_Income <= 1000 and Applicant_Income > 5000:
            income_bin_average = 0
            income_bin_high = 1
            income_bin_very_high = 0
        else:
            income_bin_average = 0
            income_bin_high = 0
            income_bin_very_high = 1
        
        income_bin_average = np.uint8(income_bin_average)
        income_bin_high = np.uint8(income_bin_high)
        income_bin_very_high = np.uint8(income_bin_very_high)

        #Co-Applicant Income
        CoApplicant_Income = request.form['CoApplicantIncome']
        CoApplicant_Income = np.float64(CoApplicant_Income)

        Total_Income = Applicant_Income + CoApplicant_Income
        Total_Income_log = np.log(Total_Income)
        Total_Income_log = np.float64(Total_Income_log)


        my_prediction = model.predict([[
            Dependents,
            Credit_History,
            loan_log,
            Gender_Male,
            Married,
            Education,
            Self_Employed,
            Property_Area_Semiurban,
            Property_Area_Urban,
            income_bin_average,
            income_bin_high,
            income_bin_very_high,
            Total_Income_log,
        ]])

        Prob = model.predict_proba([[
            Dependents,
            Credit_History,
            loan_log,
            Gender_Male,
            Married,
            Education,
            Self_Employed,
            Property_Area_Semiurban,
            Property_Area_Urban,
            income_bin_average,
            income_bin_high,
            income_bin_very_high,
            Total_Income_log,
        ]]) * 100

        no = round(Prob[0][0],2)
        yes = round(Prob[0][1],2)

    return render_template('home.html', prediction = my_prediction, No=no, Yes=yes)

if __name__ == "__main__":
    app.run(debug=True)
    
