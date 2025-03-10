https://www.kaggle.com/code/alexisbcook/introduction

### 简介
- 提升的地方
  - 处理真实数据时可能遇到的各种问题，如，数据缺失
  - 设计pipelines来提升机器学习代码的指令
  - 使用更高级的验证模型技术，如，交叉验证（cross-validation）
  - 构建新的模型--当下最好的模型（state-of-the-art model）
  - 避免常见且重要的数据科学错误，如，泄露（leakage）

### 缺失值的处理
- 很多机器学习的库，遇到缺失的值时，在构建模型时会报错
- 3个措施
  - 方案1：把整列都去掉（不建议，容易错失重要数据）
    - 直接调用数据集的drop方法
  - 方案2：进行值的填充
    - 比如，用其他非缺失值的平均值来填充
    - 更加复杂的策略，并不一定能带来额外的收获
    - 代码
      - from sklearn.impute import SimpleImputer
      - 对训练集填充：SimpleImputer.fit_transform(X_train)
      - 对测试集填充：SimpleImputer.transform(x_test)
  - 方案3：对填充的扩展
    - 增加一列，标识哪些行的数据原先是缺失的
  - 