import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from model import execute

app = Flask(__name__)

# Constants
RG_PICKLE_FILEPATH = 'models/rg.pkl'
RO_PICKLE_FILEPATH = 'models/ro.pkl'
RBSO_PICKLE_FILEPATH = 'models/rbso.pkl'

EXPECTED_COLUMNS_IN_ORDER = ['account_number_avg_order_total_per_account_number_1day','account_number_avg_order_total_per_account_number_30day','account_number_avg_order_total_per_account_number_7day','account_number_avg_order_total_per_account_number_90day','account_number_num_distinct_transaction_per_account_number_1day','account_number_num_distinct_transaction_per_account_number_30day','account_number_num_distinct_transaction_per_account_number_7day','account_number_num_distinct_transaction_per_account_number_90day','account_number_num_fraud_transactions_per_account_number_1day','account_number_num_fraud_transactions_per_account_number_30day','account_number_num_fraud_transactions_per_account_number_7day','account_number_num_fraud_transactions_per_account_number_90day','account_number_num_fraud_transactions_per_account_number_lifetime','account_number_num_order_items_per_account_number_1day','account_number_num_order_items_per_account_number_30day','account_number_num_order_items_per_account_number_7day','account_number_num_order_items_per_account_number_90day','account_number_num_order_items_per_account_number_lifetime','account_number_sum_order_total_per_account_number_1day','account_number_sum_order_total_per_account_number_30day','account_number_sum_order_total_per_account_number_7day','account_number_sum_order_total_per_account_number_90day','is_billing_shipping_city_same','is_existing_user','num_order_items','order_total','status_New','status_Pending','status_missing']


class_names = np.array(['Non-Fraudulant', 'Fraudulant'])

# def load_models():

#     rg_model_file = open(RG_PICKLE_FILEPATH, 'rb')
#     rg_model = pickle.load(rg_model_file)
#     rg_model_file.close()

#     ro_model_file = open(RO_PICKLE_FILEPATH, 'rb')
#     ro_model = pickle.load(ro_model_file)
#     ro_model_file.close()

#     rbso_model_file = open(RBSO_PICKLE_FILEPATH, 'rb')
#     rbso_model = pickle.load(rbso_model_file)
#     rbso_model_file.close()

#     return (rg_model,ro_model,rbso_model)


@app.route("/")
def display_form():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    int_features = np.array([float(x) for x in request.form.values()])

    status_values = []
    if int_features[-1] == 0:
        status_values = [1,0,0]
    elif int_features[-1] == 1:
        status_values = [0,1,0]
    elif int_features[-1] == 2:
        status_values = [0,0,1]
    
    int_features = np.concatenate((int_features[:-1],np.array(status_values)), axis=0)


    df_features = pd.DataFrame(np.reshape(int_features, (1, 29)), columns=EXPECTED_COLUMNS_IN_ORDER)

    rg, ro, rbso = execute()

    inference_generated_rules = rg.transform(X=df_features)
    inference_optimized_rules = ro.transform(X=df_features)

    inference_rules = pd.concat([inference_generated_rules, inference_optimized_rules], axis=1)

    prediction = rbso.predict(inference_rules)

    return 'The Transaction is <b> {} </b>'.format(class_names[prediction[0]])

if __name__ == "__main__":
    app.run(debug=True)