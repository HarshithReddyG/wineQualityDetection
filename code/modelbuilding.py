import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def build_models(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Store results
    predictions = {}
    accuracies = {}

    # Logistic Regression
    logistic_model = LogisticRegression(random_state=42, max_iter=500)
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)
    predictions["Logistic Regression"] = y_pred_logistic
    accuracies["Logistic Regression"] = accuracy_score(y_test, y_pred_logistic)

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    predictions["Random Forest"] = y_pred_rf
    accuracies["Random Forest"] = accuracy_score(y_test, y_pred_rf)

    # XGBoost
    xgb_model = XGBClassifier(
        random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=1, use_label_encoder=False, eval_metric="mlogloss"
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    predictions["XGBoost"] = y_pred_xgb
    accuracies["XGBoost"] = accuracy_score(y_test, y_pred_xgb)

    # LightGBM
    lgbm_model = LGBMClassifier(
        random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1,
        class_weight='balanced', n_jobs=1
    )
    lgbm_model.fit(X_train, y_train)
    y_pred_lgbm = lgbm_model.predict(X_test)
    predictions["LightGBM"] = y_pred_lgbm
    accuracies["LightGBM"] = accuracy_score(y_test, y_pred_lgbm)

    # Stacking
    stacking_model = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(random_state=42)),
            ('lgbm', lgbm_model)
        ],
        final_estimator=LogisticRegression(max_iter=500),
        cv=3,
        n_jobs=-1
    )
    stacking_model.fit(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)
    predictions["Stacking (RF + LightGBM)"] = y_pred_stack
    accuracies["Stacking (RF + LightGBM)"] = accuracy_score(y_test, y_pred_stack)

    # Return results
    return {
        "y_test": y_test,
        "predictions": predictions,
        "accuracies": accuracies
    }
