import pickle
import statistics
import time
from configparser import ConfigParser
from warnings import simplefilter

import numpy as np
import pandas as pd
import os


import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

simplefilter("ignore", category=ConvergenceWarning)

# Read config file
configur = ConfigParser()
configur.read('config.prob')

# Load language model
check_point = configur.get('parameter', 'check_point')
tokenizer = AutoTokenizer.from_pretrained(check_point)
model = AutoModel.from_pretrained(check_point, output_hidden_states=True)

#
log_prefix = configur.get('parameter', 'log_prefix')
if not os.path.exists(log_prefix):
    os.makedirs(log_prefix)
# Keep the latest
# TODO Setup ME into a project
# TODO add mode for train and evaluation
# TODO setup a separate project for this
# TODO the result into csv - with micro,macro,std-deviation and all layers **DONE
# TODO add self-similarity (Refer the paper)
# TODO add controll task of some kind
# TODO push to git

def _index_of_entity(row_index, batch_1):
    sent = batch_1.iloc[row_index]['sentence']
    entity = batch_1.iloc[row_index]['entity']
    tokenized_sent = tokenizer.encode(sent, add_special_tokens=True, max_length=100, truncation=True)
    tokenized_entity = tokenizer.encode(entity, add_special_tokens=False)
    all_index_of_entity = []
    try:
        all_index_of_entity = [tokenized_sent.index(ind) for ind in tokenized_entity]
    except ValueError as ve:
        pass
    return all_index_of_entity


def _extract_token_rep(layer_rep, batch_1):
    single_sub_token = 0
    multi_sub_token = 0
    not_found = 0
    layer_token_rep = np.zeros((layer_rep.size()[0], layer_rep.size()[2]))
    for i in range(len(layer_rep)):
        row_id = i
        ent_sub_token_ind = _index_of_entity(row_id, batch_1)
        if len(ent_sub_token_ind) == 1:
            single_sub_token += 1
            token_index = ent_sub_token_ind[0]
            # token_rep = layer_token_rep[i][token_index] Big-Mistake
            token_rep = layer_rep[i][token_index]
            layer_token_rep[i] = token_rep

        elif len(ent_sub_token_ind) > 1:
            multi_sub_token += 1
            temp_rep = [layer_rep[i][sub_token_ind] for sub_token_ind in ent_sub_token_ind]
            # Average the sub-token representations
            token_rep = torch.mean(torch.stack(temp_rep), dim=0)
            layer_token_rep[i] = token_rep
        else:
            not_found += 1
    print('       _extract_token_rep(layer_rep, batch_1)')
    print('             single_sub_token', single_sub_token)
    print('             multi_sub_token', multi_sub_token)
    print('             not_found', not_found)

    return layer_token_rep


def _evaluate_run(clf_model, test_features, test_labels):
    y_pred = clf_model.predict(test_features)
    f1_micro = f1_score(test_labels, y_pred, average='micro')
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    f1_weighted = f1_score(test_labels, y_pred, average='weighted')
    classification_rep = classification_report(test_labels, y_pred, output_dict=True)
    return y_pred, f1_micro, f1_macro, f1_weighted, classification_rep


def _save_model(_model, classifier_identifier):
    model_direct = log_prefix+'_model_out/'
    if not os.path.exists(model_direct):
        os.makedirs(model_direct)
    model_path = model_direct + classifier_identifier + '_' + '.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(_model, file)


def _test_model(test_features, test_labels, mlp_clf):
    f1_micro_ls = []
    f1_macro_ls = []
    f1_weighted_ls = []
    print('    len(test_labels)', len(test_labels))
    mlp_num_run = int(configur.get('mlp', 'mlp_num_run'))

    classification_rep = {}
    pred_per_run = []
    for i in range(mlp_num_run):
        y_pred, f1_micro, f1_macro, f1_weighted, classification_rep = _evaluate_run(mlp_clf, test_features,
                                                                                    test_labels)
        temp_pred = y_pred.tolist()
        pred_per_run.append(temp_pred)
        f1_micro_ls.append(f1_micro)
        f1_macro_ls.append(f1_macro)
        f1_weighted_ls.append(f1_weighted)
        print('             Raw f1_macro @ ', i, ' ', f1_macro)
    prediction_ls = pred_per_run[0]  # Return prediction of the first run
    mean_micro_f1 = round(statistics.mean(f1_micro_ls), 2)
    mean_macro_f1 = round(statistics.mean(f1_macro_ls), 2)
    mean_weighted_f1 = round(statistics.mean(f1_weighted_ls), 2)
    std_dev = round(statistics.stdev(f1_macro_ls), 3)
    return mean_micro_f1, mean_macro_f1, mean_weighted_f1, std_dev, prediction_ls, classification_rep


def _mlp_classifier(train_features, train_labels, test_features, test_labels, classifier_identifier):
    f1_micro_ls = []
    f1_macro_ls = []
    f1_weighted_ls = []
    print('    len(test_labels)', len(test_labels))
    mlp_num_run = int(configur.get('mlp', 'mlp_num_run'))
    mlp_iter = int(configur.get('mlp', 'mlp_iter'))
    mlp_hiddent_unit = int(configur.get('mlp', 'mlp_hiddent_unit'))
    mlp_activation = configur.get('mlp', 'mlp_activation')
    classification_rep = {}
    pred_per_run = []
    all_models = []
    for i in range(mlp_num_run):
        mlp_clf = MLPClassifier(hidden_layer_sizes=(mlp_hiddent_unit,), max_iter=mlp_iter, activation=mlp_activation,
                                random_state=i)
        mlp_clf.fit(train_features, train_labels)
        # Save the model
        all_models.append(mlp_clf)
        # _save_model(mlp_clf, classifier_identifier)
        y_pred, f1_micro, f1_macro, f1_weighted, classification_rep = _evaluate_run(mlp_clf, test_features,
                                                                                    test_labels)
        temp_pred = y_pred.tolist()
        pred_per_run.append(temp_pred)
        f1_micro_ls.append(f1_micro)
        f1_macro_ls.append(f1_macro)
        f1_weighted_ls.append(f1_weighted)
        print('             Raw f1_macro @ ', i, ' ', f1_macro)
    _save_model(all_models[0], classifier_identifier)  # Save model of the  first run
    prediction_ls = pred_per_run[0]  # Return prediction of the first run
    mean_micro_f1 = round(statistics.mean(f1_micro_ls), 2)
    mean_macro_f1 = round(statistics.mean(f1_macro_ls), 2)
    mean_weighted_f1 = round(statistics.mean(f1_weighted_ls), 2)
    std_dev = round(statistics.stdev(f1_macro_ls), 3)
    return mean_micro_f1, mean_macro_f1, mean_weighted_f1, std_dev, prediction_ls, classification_rep


def _shuffle(train_features_l0):
    print('     *** shuffle')
    print(type(train_features_l0))
    print(train_features_l0.shape)
    np.random.shuffle(train_features_l0)
    return train_features_l0


def _get_hidden_state(batch_1):
    print('  _get_hidden_state(batch_1)')
    tokenized = batch_1['sentence'].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=100, truncation=True)))
    # Padding
    pad_length = configur.get('parameter', 'pad_length')
    max_len = int(pad_length)
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

    # To ignore the padded values
    attention_mask = np.where(padded != 0, 1, 0)
    print('     attention_mask.shape ', attention_mask.shape)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    print('     Running torch.no_grad() ')
    start_time1 = time.time()
    torch.cuda.empty_cache()
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    last_hidden_states_layer_rep = last_hidden_states[2]
    print('     Finished torch.no_grad() ')
    print("     Block torch.no_grad() took %s seconds " % (time.time() - start_time1))

    labels = batch_1['class']

    return last_hidden_states_layer_rep, labels


def _expand_dic(classification_rep):
    new_dic = {}
    for k1, v1 in classification_rep.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                new_key = 'class_' + k1 + '_' + k2
                new_dic[new_key] = round(v2, 2)
    return new_dic


def main():
    layer_key = ['Layer_' + str(i) for i in range(13)]
    dic_class_report = dict.fromkeys(layer_key)
    print('layer_key ', layer_key)
    result_colmn = ['F1_micro', 'F1_macro', 'F1_weighted', 'Std_deviation']
    pd_result = pd.DataFrame(columns=result_colmn, index=layer_key)
    pd_prediction = pd.DataFrame(columns=layer_key)
    train_data_path = configur.get('parameter', 'train_data_path')
    test_data_path = configur.get('parameter', 'test_data_path')
    mode = configur.get('parameter', 'mode')
    print('mode  ', mode)
    # model_path = configur.get('parameter', 'model_path')

    # start from test file
    test_batch = pd.read_csv(test_data_path)
    print('     len(test_batch) ', len(test_batch))
    test_last_hidden_states_layer_rep, test_labels = _get_hidden_state(test_batch)
    print('test_data_path ', test_data_path)

    # last_hidden_states_layer_rep, labels = _get_hidden_state(batch_1)

    gold_labels = []
    # Process layer wise
    # if mode == 'train':
    # sample_size = int(configur.get('parameter', 'sample_size'))
    if mode == 'train':
        train_batch = pd.read_csv(train_data_path)
        print('     len(train_batch) ', len(train_batch))
        train_last_hidden_states_layer_rep, train_labels = _get_hidden_state(train_batch)
    for i in range(13):
        layer_name = 'Layer_' + str(i)
        print('  Processing layer ', i)
        # layer_rep = last_hidden_states_layer_rep[i]
        # Extract token representation from a layer i
        test_layer_rep = test_last_hidden_states_layer_rep[i]
        test_layer_token_rep = _extract_token_rep(test_layer_rep, test_batch)

        class_model_name = layer_name
        start_time1 = time.time()

        gold_labels = test_labels.tolist()

        if mode == 'train':
            train_layer_rep = train_last_hidden_states_layer_rep[i]
            train_layer_token_rep = _extract_token_rep(train_layer_rep, train_batch)
            mean_micro_f1, mean_macro_f1, mean_weighted_f1, std_dev, prediction_ls, classification_rep = _mlp_classifier(
                train_layer_token_rep,
                train_labels,
                test_layer_token_rep,
                test_labels,
                class_model_name)
        else:
            model_path = configur.get('parameter', 'model_path')
            model_full_path = model_path + layer_name + '_' + '.pkl'
            print('Evaluating ', model_full_path)
            mlp_clf = pickle.load(open(model_full_path, 'rb'))
            mean_micro_f1, mean_macro_f1, mean_weighted_f1, std_dev, prediction_ls, classification_rep = _test_model(
                test_layer_token_rep, test_labels, mlp_clf)

        dic_class_report[layer_name] = classification_rep
        pd_result.at[layer_name, 'F1_micro'] = mean_micro_f1
        pd_result.at[layer_name, 'F1_macro'] = mean_macro_f1
        pd_result.at[layer_name, 'F1_weighted'] = mean_weighted_f1

        # change classification_rep to a one level dictionary
        classification_rep = _expand_dic(classification_rep)
        for k, v in classification_rep.items():
            pd_result.at[layer_name, k] = v
        pd_result.at[layer_name, 'Std_deviation'] = std_dev
        pd_prediction[layer_name] = prediction_ls
        print("     Classification Block took %s seconds " % (time.time() - start_time1))

    pd_prediction['catagory'] = test_batch['catagory']
    # pd_prediction['Sim_DE_EN'] = test_batch['Sim_DE_EN']
    pd_prediction['Gold_Labels'] = gold_labels
    log_file_name = test_data_path.split('/')[1]
    log_file_name = log_file_name.replace('.csv', '')

    # Write Result log
    result_file_name = log_prefix +'/_result_' + log_file_name + '.csv'
    pd_result.to_csv(result_file_name)

    # Write Prediction log
    pred_file_name = log_prefix +'/_prediction_' + log_file_name + '.csv'
    pd_prediction.to_csv(pred_file_name)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Running all sample took ", (time.time() - start_time))
