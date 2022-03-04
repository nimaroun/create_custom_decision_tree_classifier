from sklearn import datasets
from sklearn.model_selection import train_test_split
from src.source_code import DecisionTreeClassifier

import_iris_df = datasets.load_iris()

iris_df = (pd.DataFrame(data= np.c_[import_iris_df['data'], import_iris_df['target']],
                        columns= import_iris_df['feature_names'] + ['target'])
          )


X = iris_df.iloc[:,:-1].values
Y = iris_df.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state = 41)

classifier = DecisionTreeClassifier(min_samples_split=3,max_depth=2)

fit_ = classifier.fit(X_train,y_train)
classifier.print_tree()

predict_ = classifier.predict(X_test)


