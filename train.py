from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
print("test score:", clf.score(X_test, y_test))
joblib.dump(clf, "model.joblib")

