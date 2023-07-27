from iguanas.rule_generation import RuleGeneratorDT
from iguanas.rule_optimisation import BayesianOptimiser
from iguanas.metrics.classification import FScore, Precision
from iguanas.metrics.pairwise import JaccardSimilarity
from iguanas.rules import Rules, ConvertProcessedConditionsToGeneral, ReturnMappings
from iguanas.correlation_reduction import AgglomerativeClusteringReducer
from iguanas.rule_selection import SimpleFilter, GreedyFilter, CorrelatedFilter
from iguanas.rbs import RBSPipeline, RBSOptimiser

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders.one_hot import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

sns.set_style('white')

def execute():

    data = pd.read_csv('data/transactions.csv', index_col='eid')

    fraud_column = 'is_fraud'
    X = data.drop(
        fraud_column, 
        axis=1
    )
    y = data[fraud_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include=object).columns.tolist()
    bool_cols = X_train.select_dtypes(include=bool).columns.tolist()

    X_train[bool_cols] = X_train[bool_cols].astype(object)
    X_test[bool_cols] = X_test[bool_cols].astype(object)

    X_train.loc[:, num_cols] = X_train.loc[:, num_cols].fillna(-1)
    X_train.loc[:, cat_cols] = X_train.loc[:, cat_cols].fillna('missing')
    X_train.loc[:, bool_cols] = X_train.loc[:, bool_cols].fillna('missing')
    X_test.loc[:, num_cols] = X_test.loc[:, num_cols].fillna(-1)
    X_test.loc[:, cat_cols] = X_test.loc[:, cat_cols].fillna('missing')
    X_test.loc[:, bool_cols] = X_test.loc[:, bool_cols].fillna('missing')

    ohe = OneHotEncoder(use_cat_names=True)

    ohe.fit(X_train)
    X_train = ohe.transform(X_train)
    X_test = ohe.transform(X_test)

    p = Precision()
    f1 = FScore(beta=1)

    params = {
        'metric': f1.fit,
        'n_total_conditions': 4,   
        'tree_ensemble': RandomForestClassifier(n_estimators=10, random_state=0),
        'target_feat_corr_types': 'Infer',
        'num_cores': 4,
        'verbose': 1
    }

    rg = RuleGeneratorDT(**params)

    X_rules_gen_train = rg.fit(
        X=X_train, 
        y=y_train
    )

    with open('data/rule_strings.pkl', 'rb') as f:
        rule_strings = pickle.load(f)

    existing_rules = Rules(rule_strings=rule_strings)
    existing_rule_lambdas = existing_rules.as_rule_lambdas(as_numpy=False, with_kwargs=True)

    params = {
        'rule_lambdas': existing_rule_lambdas,
        'lambda_kwargs': existing_rules.lambda_kwargs,
        'metric': f1.fit,
        'n_iter': 10,
        'num_cores': 4,
        'verbose': 1
    }

    ro = BayesianOptimiser(**params)

    X_rules_opt_train = ro.fit(
        X=X.loc[X_train.index], 
        y=y_train
    )

    # Combine the binary columns of each rule set
    X_rules_train = pd.concat([
        X_rules_gen_train, 
        X_rules_opt_train
    ], axis=1)


    fr = SimpleFilter(
        threshold=0.01,
        operator='>=',
        metric=f1.fit,
    )

    X_rules_train = fr.fit_transform(
        X_rules=X_rules_train, 
        y=y_train
    )

    js = JaccardSimilarity()

    acfr = AgglomerativeClusteringReducer(
        threshold=0.75,
        strategy='bottom_up', 
        similarity_function=js.fit, 
        metric=f1.fit
    )

    fcr = CorrelatedFilter(correlation_reduction_class=acfr)

    X_rules_train = fcr.fit_transform(
        X_rules=X_rules_train,
        y=y_train
    )

    gf = GreedyFilter(
        metric=f1.fit, 
        sorting_metric=p.fit,
        verbose=1
    )

    X_rules_train = gf.fit_transform(
        X_rules=X_rules_train, 
        y=y_train
    )

    config = [
        (1, X_rules_train.columns.tolist())
    ]

    final_decision = 0

    rbsp = RBSPipeline(
        config=config,
        final_decision=final_decision
    )

    rbso = RBSOptimiser(
        pipeline=rbsp, 
        metric=f1.fit,
        n_iter=60, 
        verbose=1
    )

    pipe_pred_train = rbso.fit_predict(
        X_rules=X_rules_train, 
        y=y_train
    )

    rbs_rule_names_gen = [rule for rule in rbso.rules_to_keep if rule in rg.rule_names]
    rbs_rule_names_opt = [rule for rule in rbso.rules_to_keep if rule in ro.rule_names]

    rg.filter_rules(include=rbs_rule_names_gen)
    ro.filter_rules(include=rbs_rule_names_opt)

    return (rg, ro, rbso)
