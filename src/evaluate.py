import json
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd
import xgboost


from sklearn.metrics import mean_squared_error
from sagemaker.experiments import Run


if __name__ == "__main__":
    with Run(experiment_name="localpipeline") as run:
        model_path = f"/opt/ml/processing/model/model.tar.gz" # 此處寫死模型輸出位置，不知道為何不是使用前面 Estimator 的 Output Path
        test_path = "/opt/ml/processing/test/test.csv" # 找到測試資料並讀取

        with tarfile.open(model_path) as tar: # 簡單解壓縮
            tar.extractall(path=".")
        
        run.log_parameter(name="Model Path", value=model_path)
        run.log_parameter(name="Test Data Path", value=test_path)
        
        model = pickle.load(open("xgboost-model", "rb")) # 模型讀取
        
        df = pd.read_csv(test_path, header=None)
        
        y_test = df.iloc[:, 0].to_numpy() # 把 label 欄位，變數轉換 numpy
        df.drop(df.columns[0], axis=1, inplace=True) # drop Y
        
        X_test = xgboost.DMatrix(df.values) # 應該是透過 xgboost 框架把 df x 轉換成可輸入格式
        
        predictions = model.predict(X_test) # 以 test data 進行 prediction

        mse = mean_squared_error(y_test, predictions) # MSE
        std = np.std(y_test - predictions) # 計算標準差
        report_dict = { # 定義報告格式
            "regression_metrics": {
                "mse": {
                    "value": mse,
                    "standard_deviation": std
                },
            },
        }
        run.log_metric(name="Mean Squard Error", value=mse)
        run.log_metric(name="Standard Deviation", value=std)
        output_dir = "/opt/ml/processing/evaluation" # 定義 output 位置
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) # Path 框架保證 path 的穩定性
        
        evaluation_path = f"{output_dir}/evaluation.json"
        with open(evaluation_path, "w") as f: # 寫 json file
            f.write(json.dumps(report_dict))