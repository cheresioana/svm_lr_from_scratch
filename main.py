# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pandas_profiling import ProfileReport
from scipy.stats import pearsonr

def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def plotSVM(X_test, y_test, classifier, title):
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('EstimatedSalary')
    plt.legend()
    plt.show()

def svm_technique(dataset):
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(cm)
    print("Accuracy SVC: ", accuracy_score(y_test, y_pred))
    plotSVM(X_test, y_test, classifier, "SVM classification")
    return accuracy_score(y_test, y_pred)


def logistic_regression(dataset):
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = LogisticRegression(random_state=0, solver='lbfgs')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix")
    print(cm)
    print("Accuracy LR: ", accuracy_score(y_test, y_pred))
    plotSVM(X_test, y_test, classifier, "LogisticRegression classification")
    return accuracy_score(y_test, y_pred)

# Press the green button in the gutter to run the script.
def generate_html_graph(df):
    prof = ProfileReport(df)
    prof.config.html.minify_html = False
    prof.to_file(output_file='html_stats/titanic-min.html')

def data_analysis(df):
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #generate_html_graph(df)
    #print(X)
    #print(X[:, 1])
    print("Pearson income and purchased")
    corr, _ = pearsonr(X[:, 1], y)
    print(corr)
    print("Pearson age and purchased")
    corr, _ = pearsonr(X[:, 0], y)
    print(corr)
    print("Pearson age and estimated salary")
    corr, _ = pearsonr(X[:, 0], X[:, 1])
    print(corr)


if __name__ == '__main__':
    dataset = pd.read_csv('Social_Network_Ads.csv')
    data_analysis(dataset);
    '''acc_1 = []
    acc_2 = []
    for i in range(0, 4):
        acc_1.append(svm_technique(dataset))
        acc_2.append(logistic_regression(dataset))
    print("SVM")
    print(acc_1)
    print("LR")
    print(acc_2)'''



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
