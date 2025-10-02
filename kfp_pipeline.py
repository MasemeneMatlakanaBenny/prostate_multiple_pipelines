from kfp import dsl,compiler
from kfp.dsl import Input,Output,Artifact,component

@component
def load_data(df_artifact:Output[Artifact]):
    """
    The first component for loading the dataset
    """
    #import libraries first:
    import pandas as pd

    dataset=pd.read_csv("prostate.csv")

    df_artifact.to_csv("input_data.csv")

@component
def data_split(df_artifact:Input[Artifact],
               train_artifact:Output[Artifact],
               test_artifact:Output[Artifact]):
    """
    The second component for splitting the dataset
    """
    #import libraries first:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset=pd.read_csv(df_artifact.csv)

    train_data,test_data=train_test_split(dataset,test_size=0.2,random_state=42)

    train_data.to_csv("train_data.csv")
    test_data.to_csv("test_data.csv")

@component
def model_training(train_artifact:Input[Artifact],
                   model_artifact:Output[Artifact]):
    
    """
    Use the train data obtained from the data split component to train the model.
    Save the model into an artifact using pkl
    """

    # import libraries first -> pandas and scikit-learn in this case 
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    train_df=pd.read_csv(train_artifact.csv)

    X_train=train_df.drop("Target",axis=1)
    y_train=train_df['Target']

    model=LogisticRegression(solver='liblinear')

    model.fit(X_train,y_train)

    # save the model into a pickle file-> joblib will do the work in this case:

    import joblib

    joblib.dump(model,"model_prostate.pkl")


@component
def model_eval(test_artifact:Input[Artifact],
               model_artifact:Input[Artifact]):
    
    """
    Component for evaluating the model
    The following key metrics are used -> accuracy,cohen_kappa_score 
    and matthews correlation coefficient.
    """

    # import libs -> pandas and key metrics from sklearn

    import pandas as pd
    from sklearn.metrics import cohen_kappa_score,matthews_corrcoef,accuracy_score

    test_data=pd.read_csv(test_artifact.csv)

    X_test=test_data.drop("Target",axis=1)
    y_test=test_data["Target"]

    # reload the model -> joblib since it was used to save the model in the first place:
    import joblib
    model=joblib.load("model_prostate.pkl")

    y_preds=model.predict(X_test)

    return {"accuracy score":accuracy_score(y_test,y_preds),
            "kappa score":cohen_kappa_score(y_test,y_preds),
            "mat score":matthews_corrcoef(y_test,y_preds)}
 
