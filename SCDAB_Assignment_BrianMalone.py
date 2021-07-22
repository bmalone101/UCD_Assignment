import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import easygui
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import PySimpleGUI as sg
import re
from tkinter import Tk, filedialog

#2021721 Final Version - Brian Malone 22 July 2021

# 20210630: Dataset is form https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset/tasks but joined with Random Names
# & Phone Numbers which is a different Dataset created by Brian Malone (All comments are dated as YYYYMMDD)

# 20210717: Setting the GUI theme for my program
sg.theme('Dark Blue')

#Starting the GUI with a welcome and request
sg.popup('Welcome to the Framingham Health Screening Model', 'Press OK\nThen select your home directory, ensuring all input files are located there')

#20210720 Using tkinter to select the home directory to be used in the code. I found this method the easiest.
root = Tk()
root.withdraw()
root.attributes('-topmost', True)
home_folder = filedialog.askdirectory()
os.chdir(home_folder)

# 20210717 This pnumber() code is added to demonstrate how to use REGEX and pick a pattern out of a .csv file
def pnumber():
    with open('FrameComplete.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        clist = df4['Phone'].tolist()
        rate = ''.join(map(str,clist))
        sg.popup_ok('Welcome - This part of the program will allow you to extract either 087 or 086 numbers')
        colum = easygui.enterbox("What number do you want to extract: Either 087 or 086")
        #Finding phone numbers of people with 086/087
        if colum == '087':
            high = re.findall(r"087-\d\d\d\d\d\d\d", rate)
            voda = open('Zero87.csv', "w")
            wr=csv.writer(voda,delimiter="\n")
            wr.writerow(high)
        elif colum == '086':
            high = re.findall(r"086-\d\d\d\d\d\d\d", rate)
            voda = open('Zero86.csv', "w")
            wr = csv.writer(voda, delimiter="\n")
            wr.writerow(high)
        else:
            easygui.msgbox('[There are only 087 and 086 numbers]', 'ERROR ERROR ERROR')

#20210717 The covar() function was added to demonstrate the use of Numpy functions.
def covar():
    sg.popup_ok('This part will perform a covariance calculation of the variables', title='Welcome')
    layout = [[sg.Text('Calculate Correlation Coefficient two values: (Put any two values into the input boxes)')],
              [sg.Text('education \ncurrentSmoker \nCigsPerDay \nBPMeds \nBMI \nsysBP \ndiaBP \n male'), sg.InputText()],
              [sg.Text( 'prevalentHyp \ndiabetes \ntotChol \nprevalentStroke \nHeartRate \nglucose \n age'), sg.InputText()], [sg.Submit(), sg.Cancel()]]
    window = sg.Window('Covariance Inputs', layout, size=(550, 340))
    while True:
        event, values = window.read()
        val1 = values[0]
        #Taking the user inputs
        val2 = values[1]
        #Taking the user input

        # pulling the corresponding val1 column into a numpy array to be compared
        cof1 = np.asarray(df4[val1])
        # pulling the corresponding val2 column into a numpy array to be compared
        cof2 = np.asarray(df4[val2])
        # Finding the Pearson Covariance matrix for the values.
        cof = np.corrcoef(cof1,cof2)
        # Extract the covariance from the matrix
        cov = cof[0, 1]
        print(cov)
        # Change to string since pop can't use numpy float
        cov = str(round(cov, 4))
        # Output showing the variance.
        #covariance = np.cov(cof1,cof2)
        # This value will be used to put value dependent colour on the scatter plot based on cof2 (y-axis)
        t = cof2
        #Code in order to plot the trend line
        b, m = polyfit(cof1, cof2, 1)
        plt.scatter(cof1,cof2, c=t, cmap='viridis')
        plt.colorbar()
        plt.plot(cof1, b + m * cof1, '-')
        plt.title('Correlation Coefficient Scatter Plot' + "CorrCoef =" + cov)
        #Label the Plot
        plt.xlabel(val1)
        plt.ylabel(val2)
        plt.show()

        sg.popup_ok('Correlation Coefficient is  ' + cov + '.', title='Output')

        if event == sg.WIN_CLOSED or event == 'Cancel':
            break
        break
    window.close()

# 20210629: Standalone function that will clean data (if needed)
def clean(file):
    #Made df4 global since it is used throughout the code.
    global df4
    #Removing NA with mean rounded to 2
    df4 = file.fillna(round(file.mean(),2))
    df4.to_csv('FrameComplete.csv')

# 20210613: Separating Phone Numbers into .csv for male and for female
# Putting the Dataframe with the Phone Numbers and References into a Dictionary
def SeparateMaleFemale():
    event = sg.popup_yes_no('Would you like to extract Male and Female Contacts into separate files')
    if event == 'Yes':
        #Open the file that we want to loo inside
        with open('FrameComplete.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # Creating the new contact spreadsheets
            m = open('MaleContacts.csv', "w")
            f = open('femaleContacts.csv', "w")
            # Outlining the fieldnames that we wish to pull out.
            fieldnames = ('Phone', 'GivenName', 'Surname')
            writer = csv.DictWriter(m, fieldnames=fieldnames, lineterminator = '\n')
            writer.writeheader()
            writerf = csv.DictWriter(f, fieldnames=fieldnames,lineterminator = '\n')
            writerf.writeheader()
            # Pullout the Phone Number and Given name associated with the gender selected.
            for row in reader:
                            if row['Gender_x'] == 'male':
                                writer.writerow({'Phone': row['Phone'], 'GivenName': row['GivenName'], 'Surname': row['Surname']})
                            else:
                                writerf.writerow({'Phone': row['Phone'], 'GivenName': row['GivenName'], 'Surname': row['Surname']})
    else:
        sg.popup_ok('OK - We will skip this')

# 20210630 Function to create Query.csv that pulls data above certain value
def rate(file,column):
    with open(file) as csvfile:
        cv = pd.read_csv(csvfile)
        hrq = open('RateQuery.csv', "w")
        # Cleaning the data in the case that it is needed.
        cv[column] = cv[column].fillna(round(cv[column].mean(), 2))
        # GUI to take input from User, i use easygui which i found useful for taking user input easily
        test = easygui.enterbox("Above what threshold value do you want to pull data for?:")
        # Change the input from str to int
        test = int(test)
        # setup the header for the output file
        header = ['GivenName','Surname', column]
        # setup the writing to external CSV
        right = csv.writer(hrq, lineterminator = '\n')
        # writing header to output file
        right.writerow(header)
        #Iterating through the rows of the dataframe and outputting the Names of people above the specified Threshold
        for index, row in cv.iterrows():
            #Changing to a float so that it works in an if statement.
            row[column] = int(float(row[column]))
            if row[column] > test:
                # These are the values we are reporting on:
                p = [row['GivenName'], row['Surname'], row[column]]
                #right.writerow(header)
                right.writerow(p)
                #right.writerow({row['heartRate']})
            #f.close()



# 20200701: ****Tree Decision Testing****
# Risk is the 10 year Health risk that is being investigated, Inputs are Binary
def tree_model(dataframe,Risk,Input1,Input2,Input3):
    #Defining the training data
    training=pd.get_dummies(dataframe, columns=[Input1,Input2,Input3])
    # Creating a list of data columns to be dropped rather than doing them one by one
    drop_list = ["Title", "Phone", "Surname", "GivenName","Gender_x", "Gender_y"]
    training.drop(drop_list,axis = 1,inplace=True)
    print(training.head())
    # Different sets of independent and dependent variables
    health_x=training.drop(Risk,axis=1)
    health_y=training[Risk]
    print(health_y.shape)
    print(health_x.shape)

    #20210101: Split into Training and Test datasets
    health_x_train, health_x_test, health_y_train, health_y_test = train_test_split(health_x, health_y, test_size=0.30, random_state=0)
    clf = tree.DecisionTreeClassifier(criterion='gini',min_samples_leaf=40)
    clf=clf.fit(health_x_train, health_y_train)
    predictions = clf.predict(health_x_test)
    # Making score global since I want to transfer this to a output display
    global score
    score = clf.score(health_x_test, health_y_test)
    # What is the accuracy of the model?
    print(score)
    cm = metrics.confusion_matrix(health_y_test, predictions)
    print(cm)

    #Plotting the Matrix
    sg.popup_ok('Next you will see a graph of the Tree-Based Predictions')
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Tree Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.show()
    # 20210716 Finding out what variables influence CVD the most.
    Feat_Imp = pd.DataFrame({"Features": health_x_train.columns, "Importance": clf.feature_importances_})
    Feat_Imp = Feat_Imp.sort_values(by=['Importance'], ascending=False)
    # There are too many factors in Feat_Imp so i have taken out the top 10 and plotted them
    Feat = Feat_Imp.head(10)

    event1 = sg.popup_yes_no('Would you like the ten most important factors')
    if event1 == 'Yes':
        sg.popup_ok(Feat)
        plotty = sns.barplot(x="Features", y="Importance", data=Feat)
        plt.show()
    else:
        sg.popup_ok('OK - We will skip this')

    true_negative = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_positive = cm[1][1]

    #20210716 Doing some Boosting now;
    sg.popup_ok('The next will graph the results from Boosting')
    from sklearn.ensemble import GradientBoostingClassifier
    clf_boost = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    clf_boost = clf_boost.fit(health_x_train, health_y_train)

    predictionsb = clf_boost.predict(health_x_test)
    print('\nTarget on test data', predictionsb)

    scoreb = clf_boost.score(health_x_test, health_y_test)
    print(scoreb)

    cmb = metrics.confusion_matrix(health_y_test, predictionsb)
    print(cmb)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cmb, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Boosting Accuracy Score: {0}'.format(scoreb)
    plt.title(all_sample_title, size=15)
    plt.show()

    #20210716 Going to preform Logistic Regression with the same dataset
    #Create a new train/test split for the logical test with its own parameters
    sg.popup_ok('The next will graph the results from Logistic Regression')
    healthl_x_train, healthl_x_test, healthl_y_train, healthl_y_test = train_test_split(health_x, health_y,test_size=0.25,random_state=0)

    logisticRegr = LogisticRegression(solver='newton-cg', max_iter=3000)
    logisticRegr.fit(healthl_x_train, healthl_y_train)
    predictionslog = logisticRegr.predict(healthl_x_test)
    scorelog = logisticRegr.score(healthl_x_test, healthl_y_test)
    print(scorelog)

    cmlog = metrics.confusion_matrix(healthl_y_test, predictionslog)
    print(cmlog)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cmlog, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Regression Accuracy Score: {0}'.format(scorelog)
    plt.title(all_sample_title, size=15)
    plt.show()

    # 20210701: Printing the various values which indicate how good the model was, starting with Accuracy
    global Accuracy
    Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    print('Accuracy is:')
    print(Accuracy)
    print('')
    #Misclassification
    global Misclassification_rate
    Misclassification_rate = (false_positive + false_negative) / (true_positive + false_positive + false_negative + true_negative)
    print('Misclassification is:')
    print(Misclassification_rate)
    print('')
    #Precision is made Global so that I can send it to the GUI
    global Precision
    Precision = true_positive / (true_positive + false_positive)
    print('Precision is:')
    print(Precision)
    print('')
    global Recall # Recall is made global so that I can send it to the GUI
    Recall = true_positive / (true_positive + false_negative)
    print('Recall is:')
    print(Recall)
    print('')
    #F1
    global F1_Score
    F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
    print('F1 is:')
    print(round(F1_Score,2))
    print('')
    return(cm,Precision,Recall,Misclassification_rate,F1_Score)

# 20210702: The callback function for Precision
def button1():
    print('Precision is')
    print(Precision)
    sg.popup_ok(round(Precision,3))

# 20210702: callback function for Recall
def button2():
    print('Recall is')
    print(round(Recall,3))
    sg.popup_ok(round(Recall,3))

# 20210702: callback function for Accuracy
def button3():
    print('Accuracy')
    print(round(Accuracy,3))
    sg.popup_ok(round(Accuracy,3))

# 20210702: callback function for Misclassification
def button4():
    print('Misclassification Rate is')
    print(round(Misclassification_rate,3))
    sg.popup_ok(round(Misclassification_rate,3))

# 20210702: callback function for F1 Score
def button5():
    print('F1 Score is')
    print(F1_Score)
    sg.popup_ok(round(F1_Score,3))

def button6():
    print('CM is')
    print(score)
    sg.popup_ok(round(score,3))

#20210717 Calling the various functions:

# File with all the Health Details but no names
df = pd.read_csv("framingham.csv", encoding="ISO-8859-1")
# File with all the Names and Phone numbers
df2 = pd.read_csv("Frame_numbers.csv", encoding="ISO-8859-1")

#20210629: Cleaning the NA in the glucose column and replacing with mean() for the same column, rounded to nearest number
df['glucose'] = df['glucose'].fillna(round(df['glucose'].mean(),2))

#202106 Writing the new data to a new file in the same location directory
df.to_csv('FrameGlucoseClean.csv') #File with clean glucose

#202106 Joining the dataframe with health details with the one with contact details
df3=pd.merge(df,df2,on="Phone")
#Function to clean the dataframe of null values, i send the merged dataframe to this function.
clean(df3)

# Function that will separate the male and female participants of the study into separate .csv files
SeparateMaleFemale()
# Function that will pull out either 087 or 086 numbers from the database.
pnumber()
# Function to calculate teh covariance between two variables in the study
covar()
sg.popup_ok('The next will allow you to extract contacts with BMI, HeartRate or CigsPerDay above any threshold')
# GUI to take input from User
colum = easygui.enterbox("What column are you testing for? BMI or HeartRate or CigsPerDay:")
# Function that allows you to pull either the BMI or Heart Rate above a user inputted threshold
rate('FrameComplete.csv',colum) # Send value to be extracted to the rate function

# 20210701 This is the function for modelling, 1st is the dataframe, next is the risk being modelled & the last four are the columns used for training and testing.
tree_model(df4,"TenYearCHD","currentSmoker","prevalentStroke",'prevalentHyp')
# 20210702: Lookup dictionary that maps button to function to call
dispatch_dictionary = {'Precision':button1, 'Recall':button2, 'Accuracy':button3, 'Misclassification':button4, 'F1':button5, 'Tree Score':button6}

# 20210702 Layout the design of the GUI
layout = [[sg.Text('Please click a button', auto_size_text=True)],
          [sg.Button('Precision'), sg.Button('Recall'), sg.Button('Accuracy'), sg.Button('Misclassification'),sg.Button('F1'), sg.Button('Tree Score'), sg.Quit()]]

# 20210702 Show the Window to the user
window = sg.Window('Tree Model Properties -Original Tree Test', layout) # Name of the Box

# Event loop. Read buttons, make callbacks
while True:
    # Read the Window
    event, value = window.read()
    if event in ('Quit', sg.WIN_CLOSED):
        break
    # Lookup event in function dictionary
    if event in dispatch_dictionary:
        # get function from dispatch dictionary
        func_to_call = dispatch_dictionary[event]

        func_to_call()
    else:
        print('Event {} not in dispatch dictionary'.format(event))

window.close()
# All done!
sg.popup_ok('Bye Bye - Thanks for the help UCD')