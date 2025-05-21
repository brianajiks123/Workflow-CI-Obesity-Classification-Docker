import argparse, pandas as pd, mlflow, mlflow.sklearn, joblib, pathlib

from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from mlflow.models.signature import infer_signature


def setup_mlflow(mode, local_uri=None, repo_owner=None, repo_name=None):
    if mode == 'local':
        if not local_uri:
            raise ValueError("--local_uri must be filled on local mode!")
        
        mlflow.set_tracking_uri(local_uri)
        
        print(f"✅ MLflow mode: LOCAL ({local_uri})")
    elif mode == 'online':
        if not (repo_owner and repo_name):
            raise ValueError("--repo_owner and --repo_name must be filled for online mode!")
        
        import dagshub
        
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        
        print(f"✅ MLflow mode: ONLINE via DagsHub ({repo_owner}/{repo_name})")
    else:
        raise ValueError("Mode must be 'local' or 'online!'")


def tune_and_log(name, pipeline, param_grid, X_train, X_test, y_train, y_test, cv):
    best_f1 = -1
    best_model = None
    best_params = None
    best_run_id = None

    for i, params in enumerate(ParameterGrid(param_grid)):
        pipe = clone(pipeline)
        pipe.set_params(**params)
        
        with mlflow.start_run(run_name=f"{name}_trial_{i}") as run:
            mlflow.log_params(params)

            # Cross-validation
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            
            mlflow.log_metric("cv_f1_weighted", scores.mean())

            # Fit & evaluate
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
            mlflow.log_metric("f1", f1_score(y_test, y_pred, average='weighted'))

            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_test)
                
                mlflow.log_metric("roc_auc_ovr", roc_auc_score(y_test, proba, multi_class='ovr'))
                mlflow.log_metric("log_loss", log_loss(y_test, proba))

            signature = infer_signature(X_train, pipe.predict(X_train))
            
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=signature,
                input_example=X_train[:5]
            )

            current_f1 = f1_score(y_test, y_pred, average='weighted')
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model = pipe
                best_params = params
                best_run_id = run.info.run_id

    print(f"\n== Summary ({name}) ==")
    print(f"Best params: {best_params}")
    print(f"Best test F1: {best_f1:.3f}")
    
    return best_model, best_f1, best_run_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', required=True, help='Path to Obesity_preprocessing.csv')
    parser.add_argument('--mode', choices=['local', 'online'], default='local')
    parser.add_argument('--local_uri', help='local URI MLflow')
    parser.add_argument('--repo_owner', help='Owner DagsHub repository')
    parser.add_argument('--repo_name', help='DagsHub repository name')
    parser.add_argument('--experiment_name', default='AllRuns_CI', help='MLflow experiment name')
    
    args = parser.parse_args()

    # Setup MLflow & experiment
    setup_mlflow(args.mode, args.local_uri, args.repo_owner, args.repo_name)
    
    mlflow.set_experiment(args.experiment_name)

    # Load data
    df = pd.read_csv(args.input)
    X = df.drop('Obesity', axis=1)
    y = df['Obesity']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model configurations
    configs = [
        (
            'RandomForest',
            ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
            ]),
            {'rf__n_estimators': [100, 200], 'rf__max_depth': [10, 20, None]}
        ),
        (
            'GradientBoosting',
            ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42))
            ]),
            {'gb__n_estimators': [100, 200], 'gb__learning_rate': [0.05, 0.1], 'gb__max_depth': [3, 5]}
        ),
        (
            'XGBoost',
            ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('xgb', XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, verbosity=0, random_state=42))
            ]),
            {'xgb__n_estimators': [100, 200], 'xgb__learning_rate': [0.05, 0.1], 'xgb__max_depth': [3, 5]}
        ),
        (
            'MLP',
            ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('mlp', MLPClassifier(max_iter=1000, early_stopping=True, tol=1e-4, random_state=42))
            ]),
            {'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'mlp__alpha': [1e-4, 1e-3]}
        ),
        (
            'LogisticRegression',
            ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
            ]),
            {'lr__C': [0.1, 1.0, 10.0]}
        )
    ]

    # Tuning & logging
    best_model, best_f1, best_run_id = None, -1, None
    
    for name, pipe, grid in configs:
        m, f1, run_id = tune_and_log(name, pipe, grid, X_train, X_test, y_train, y_test, cv)
        
        if f1 > best_f1:
            best_model, best_f1, best_run_id = m, f1, run_id

    # Save best_model
    joblib.dump(best_model, "best_model.pkl")

    # Log best model to new run
    with mlflow.start_run(run_name=f"BestModel_{best_run_id}"):
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=infer_signature(X_train, best_model.predict(X_train)),
            input_example=X_train[:5],
            conda_env=str(pathlib.Path(__file__).parent / "conda.yaml")
        )
        mlflow.log_param("best_model_run", best_run_id)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.set_tag("best_model", "true")

    print(f"\n🏆 Best model run: {best_run_id}, F1-weighted: {best_f1:.3f}")
