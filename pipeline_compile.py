import kfp
from kfp import dsl
from kfp.dsl import Output, Artifact, Model, Metrics

# Component A: Load and Preprocess Data
@dsl.component(base_image='python:3.9')
def load_data_component(dataset_name: str, data_output: Output[Artifact]):
    from sklearn.datasets import load_iris
    import json
    
    iris = load_iris()
    dataset = {
        "data": iris.data.tolist(),
        "target": iris.target.tolist(),
        "feature_names": iris.feature_names
    }
    with open(data_output.path, 'w') as f:
        json.dump(dataset, f)
    print("Component A: Data Loaded and Preprocessed successfully.")

# Component B: Train Model (Produces artifacts)
@dsl.component(base_image='python:3.9')
def train_model_component(data_input: Artifact, model_output: Output[Model]):
    import json
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    
    with open(data_input.path, 'r') as f:
        dataset = json.load(f)
        
    X = dataset["data"]
    y = dataset["target"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open(model_output.path, 'wb') as f:
        pickle.dump(model, f)
    print("Component B: Random Forest Model Trained and Saved.")

# Component C: Evaluate Model (Captures metrics)
@dsl.component(base_image='python:3.9')
def evaluate_model_component(data_input: Artifact, model_input: Model, metrics_output: Output[Metrics]):
    import json
    import pickle
    
    with open(data_input.path, 'r') as f:
        dataset = json.load(f)
    X = dataset["data"]
    y = dataset["target"]
    
    with open(model_input.path, 'rb') as f:
        model = pickle.load(f)
        
    accuracy = model.score(X, y)
    metrics_output.log_metric('accuracy', float(accuracy))
    print(f"Component C: Evaluation Complete. Test Accuracy: {accuracy}")

# Kubeflow Pipeline Orchestration Definition
@dsl.pipeline(
    name='mlops-iris-end-to-end-pipeline',
    description='Automated Pipeline: Data Preprocessing, Model Training, and Evaluation'
)
def iris_pipeline():
    # Execution Flow Mapping
    load_task = load_data_component(dataset_name="iris")
    train_task = train_model_component(data_input=load_task.outputs['data_output'])
    evaluate_task = evaluate_model_component(
        data_input=load_task.outputs['data_output'], 
        model_input=train_task.outputs['model_output']
    )

# Compile Pipeline to .yaml format
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path='pipeline.yaml'
    )
    print("SUCCESS: pipeline.yaml compiled successfully!")