import os

import pandas as pd 
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# next step func
def go_next_step():

    """
    Pause execution until the user presses Enter, then clear the terminal.

    Prints a separator line, prompts the user to continue, and clears the
    terminal screen using the appropriate command for the current operating
    system.

    Returns:
        None
    """

    print("---------------")
    input("Press enter to go next -> ")
    os.system("cls" if os.name == "nt" else "clear")

# reading data
path = "bezdekIris.data"
with open(path, "r") as file:
    raw_data = file.read()

raw_data = raw_data.splitlines()
raw_data = [data.split(",") for data in raw_data]

pdf = pd.DataFrame(raw_data)    
pdf.columns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
pdf[["sepal-length", "sepal-width", "petal-length", "petal-width"]] = pdf[["sepal-length", "sepal-width", "petal-length", "petal-width"]].apply(pd.to_numeric, errors="coerce")

# some system details
print("Current Directory\n---------------\n", os.getcwd())
go_next_step()

print("List of directories\n---------------\n", os.listdir())
go_next_step()

# data sumerize
print("Shape of the dataframe\n---------------\n", pdf.shape)
go_next_step()

print("Head of the dataframe\n---------------\n", pdf.head())
go_next_step()

print("Describe of the dataframe\n---------------\n", pdf.describe())
go_next_step()

print("Size of the class column\n---------------\n", pdf["class"].value_counts())
go_next_step()

print("Data tyoes\n---------------\n", pdf.dtypes)
go_next_step()

# data visualization
pdf.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()
go_next_step()

pdf.hist()
plt.show()
go_next_step()

scatter_matrix(pdf)
plt.show()
go_next_step()

# split data
array = np.asanyarray(pdf)
X = array[:, 0:4]
y = array[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# build models
models = []
models.append(("LR", LogisticRegression(solver="liblinear")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC()))

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold.get_n_splits(), scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean()} {cv_results.std()}")
go_next_step()

plt.boxplot(results, label=names)
plt.title("Algorith comparison")
plt.show()
go_next_step()



