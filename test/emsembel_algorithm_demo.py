from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# 创建测试数据
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'yo')
plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bs')
plt.show()

# 选择分类算法，定义模型
svc_model = SVC(random_state=42)
logistic_model = LogisticRegression(random_state=42)
random_model = RandomForestClassifier(random_state=42)

# 定义投票器
voting_model = VotingClassifier(
    estimators=[
        ("svc", svc_model),
        ("logistic", logistic_model),
        ("random", random_model)
    ],
    voting="hard"
)

# 打印每个分类器的结果，及硬投票的结果
for model in [svc_model, logistic_model, random_model, voting_model]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__}: {accuracy_score(y_test, y_pred)}")

# 软投票
svc_model_2 = SVC(probability=True, random_state=42)
logistic_model_2 = LogisticRegression(random_state=42)
random_model_2 = RandomForestClassifier(random_state=42)

voting_model_2 = VotingClassifier(
    estimators=[
        ("svc", svc_model_2),
        ("logistic", logistic_model_2),
        ("random", random_model_2)
    ],
    voting="soft"
)

for model in [svc_model_2, logistic_model_2, random_model_2, voting_model_2]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__}: {accuracy_score(y_test, y_pred)}")

