#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import configparser
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    parser = argparse.ArgumentParser(description='模型训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    feature_dir = config.get('features', 'feature_dir')
    model_dir = config.get('models', 'model_dir')
    model_type = config.get('models', 'model_type', fallback='RandomForest')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for filename in os.listdir(feature_dir):
        if filename.endswith('_features.csv'):
            stock_code = filename.split('_features.csv')[0]
            print(f"正在为 {stock_code} 训练模型...")

            feature_path = os.path.join(feature_dir, filename)
            df = pd.read_csv(feature_path).dropna()

            if df.empty:
                print(f"{stock_code} 的特征数据为空，跳过训练。")
                continue

            # 创建标签：如果未来5天收盘价上涨则为1，否则为0
            df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
            df = df.dropna()

            features = ['ma5', 'ma10', 'rsi', 'macd', 'signal_line']
            X = df[features]
            y = df['target']

            if len(X) < 20: # 数据太少无法有效训练和测试
                print(f"{stock_code} 的数据不足，跳过训练。")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

            model = None
            if model_type == 'RandomForest':
                model = train_random_forest(X_train, y_train)
            else:
                print(f"不支持的模型类型: {model_type}")
                continue

            # 评估模型
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{stock_code} 的模型在测试集上的准确率: {accuracy:.4f}")

            # 保存模型
            model_path = os.path.join(model_dir, f"{stock_code}_model.joblib")
            joblib.dump(model, model_path)
            print(f"{stock_code} 的模型已保存至 {model_path}")

if __name__ == '__main__':
    main()
