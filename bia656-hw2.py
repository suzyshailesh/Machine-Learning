from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

def read_csv(file_path):
    #read in csv file
    csv_file = file_path
    df = pd.read_csv(csv_file, delimiter=';')

    #clean up dataframe
    #drop variables that do not affect outcome
    #convert categorical variables into dummy booleans
    df = df.drop(["day", "month"], axis='columns')
    for var in ["default", "housing", "loan", "y"]:
        df[var] = df[var].map(
                        {'yes':True ,'no':False})
    cat = ["job", "marital", "education", "contact", "poutcome"]
    df = pd.get_dummies(df, columns=cat)
        
    #display the DataFrame
    print(df.head())

    #split df into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    #assign the target variable "y"
    x_train = train_df.drop("y", axis=1)
    y_train = train_df["y"]
    x_test = test_df.drop("y", axis=1)
    y_test = test_df["y"]

    return x_train, y_train, x_test, y_test

def train_model(model_name, model, x_train, y_train, x_test, y_test):
    #fit model and predict values for the features test set
    model.fit(x_train, y_train)
    model_pred = model.predi
    
    #calculate and print accuracy score
    model_accuracy = accuracy_score(y_test, model_pred)
    print(model_name + " Score:", model_accuracy)

    #get roc curve and precision recall curve info
    model_roc_fpr, model_roc_tpr, _ = roc_curve(y_test, model_pred)
    model_prc_precision, model_prc_recall, _ = precision_recall_curve(y_test, model_pred)

    return model_roc_fpr, model_roc_tpr, model_prc_precision, model_prc_recall


def main():
    #create pandas df from csv file 
    #and convert to train and test sets
    x_train, y_train, x_test, y_test = read_csv("/Users/susmithashailesh/Desktop/bia652/bank.csv")

    #logistic regression
    logistic_regression = LogisticRegression()
    logistic_regression_fpr, logistic_regression_tpr, logistic_regression_precision, logistic_regression_recall = train_model("Logistic Regression", logistic_regression, x_train, y_train, x_test, y_test)

    #decision tree
    decision_tree = DecisionTreeClassifier()
    decision_tree_fpr, decision_tree_tpr, decision_tree_precision, decision_tree_recall = train_model("Decision Tree", decision_tree, x_train, y_train, x_test, y_test)

    #random forest
    random_forest = RandomForestClassifier()
    random_forest_fpr, random_forest_tpr, random_forest_precision, random_forest_recall = train_model("Random Forest", random_forest, x_train, y_train, x_test, y_test)

    #svc
    svc = SVC()
    svc_fpr, svc_tpr, svc_precision, svc_recall = train_model("SVC", svc, x_train, y_train, x_test, y_test)

    #plot roc curve
    plt.plot(logistic_regression_fpr, logistic_regression_tpr, label="Logistic Regression")
    plt.plot(decision_tree_fpr, decision_tree_tpr, label="Decision Tree")
    plt.plot(random_forest_fpr, random_forest_tpr, label="Random Forest")
    plt.plot(svc_fpr, svc_tpr, label="Support Vector Classifier")
    plt.title('ROC Curve')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend()
    plt.show()

    #plot precision recall curve
    plt.plot(logistic_regression_recall, logistic_regression_precision, label="Logistic Regression")
    plt.plot(decision_tree_recall, decision_tree_precision, label="Decision Tree")
    plt.plot(random_forest_recall, random_forest_precision, label="Random Forest")
    plt.plot(svc_recall, svc_precision, label="Support Vector Machine")
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    #print out conclusions drawn
    print("Conclusions:")
    print("1. All of the  classifiers had similar accuracy scores, all consistently falling above 0.85. These are all pretty high accuracy scores, " + \
          "showing that they are all effective at predicting if the client will subscribe to the term deposit." + \
          " The Random Forest Classifier is relatively the strongest model, with an accuracy score of 0.896.")
    print("2. Based on the ROC Curve, we can identify the Decision Tree Classifier as being the strongest model and the Support " + \
          "Vector Classifier as being the weakest. Aside from the Support Vector Classifier, the other models fall in the threshold " + \
          "for being a good model for making this prediction.")
    print("3. The Precision Recall Curve also shows the Decision Tree Classifier as the best and the Support Vector Classifier " + \
          "as the worst. However, none of the models look like great performers on this graph.")

if __name__ == "__main__":
    main()