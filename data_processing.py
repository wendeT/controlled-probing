import re
import pandas as pd
import pickle

SENT_FILE_PATH = ['xlent/en-nl/en-nl.sents.en', 'xlent/en-nl/en-nl.sents.nl', 'xlent/en-de/en-de.sents.de',
                  'xlent/en-am/en-am.sents.am', 'xlent/en-ar/en-ar.sents.ar']
TAG_FILE_PATH = ['xlent/en-nl/en-nl.tags.en', 'xlent/en-nl/en-nl.tags.nl', 'xlent/en-de/en-de.tags.de',
                 'xlent/en-am/en-am.tags.am', 'xlent/en-ar/en-ar.tags.ar']
FILTERED_TYPES = ['B-PERSON', 'B-ORG', 'B-LOC', 'B-EVENT']
LANG_ID = ['EN', 'NL', 'DE', 'AM', 'AR']
WRITE_PATH = 'xlent/filtered_en_nl_de_am_ar/'
CATAGORY = ['TRUE_MONO', 'TRUE_POLY', 'UNCLASSIFIED']


# Create a dataframe - entity, ent-type, frquency, sent_id

# Filters1 - one entity per senetnce
# Filters2 - entity type one of FILTERED_TYPES
# Filters3 - Single entity
# Filter4 - Assign

# TODO


def save_to_pickle(_data, _file_name, lang_id):
    file_path = WRITE_PATH + lang_id + '_' + _file_name + '.pickle'
    with open(file_path, 'wb') as handle:
        pickle.dump(_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Pickle saved to ', file_path)


def is_single_word_entity(label_sequence):
    pattern1 = '|'.join(FILTERED_TYPES)  # Filter1
    pattern2 = 'B-|I-'  # Filter2
    result1 = re.findall(pattern1, label_sequence)
    result2 = re.findall(pattern2, label_sequence)
    return [True, result1] if (len(result1) == 1 and len(result2) == 1) else [False, result1]


def read_data(_path):
    dic_data = {}
    with open(_path) as file:
        for sent_id, line in enumerate(file):
            dic_data[sent_id] = line
    return dic_data


def normalize_distro(row_type_distro):
    total_frq = sum(row_type_distro)
    normalized_distro = [round(float(i / total_frq), 2) for i in row_type_distro]
    return normalized_distro


def add_label(filtered_pd):
    # Add one of the catagory from CATAGORY
    catagory_ls = []
    Total_frq = []
    # print(filtered_pd.head())
    # for ind in filtered_pd.index:
    # for ind in range(len(filtered_pd)):
    for index, row in filtered_pd.iterrows():
        raw_type_distro = row.tolist()
        # print(row[FILTERED_TYPES[0]])
        normalized_type_distro = normalize_distro(raw_type_distro)
        max_value = max(normalized_type_distro)
        index_of_max_value = normalized_type_distro.index(max_value)
        # print('raw_type_distro ', raw_type_distro)
        # print('normalized_type_distro ', normalized_type_distro)
        Total_frq.append(sum(raw_type_distro))
        if raw_type_distro.count(0) == 3 or max_value >= 0.90:
            temp_label_name = CATAGORY[0] + '_' + str(FILTERED_TYPES[index_of_max_value])
            catagory_ls.append(temp_label_name)
        elif max_value <= 0.6:
            catagory_ls.append(CATAGORY[1])
        else:
            catagory_ls.append(CATAGORY[2])
    filtered_pd['Frequency'] = Total_frq
    filtered_pd['Catagory'] = catagory_ls
    # sort by frequency
    filtered_pd.sort_values(by='Frequency', ascending=False, inplace=True)

    return filtered_pd


def create_df_list(lang_id_ls, filtered_sent_id, filtered_entity_ls, filtered_entity_type_ls):
    filtered_pd = pd.DataFrame(
        list(zip(lang_id_ls, filtered_sent_id, filtered_entity_ls, filtered_entity_type_ls)),
        columns=['Lang', 'Sent Id', 'Entity', 'Entity Type'])

    file_name = str(lang_id_ls[0]) + '_' + 'entity_to_type_map' + '.csv'
    file_path = WRITE_PATH + file_name
    filtered_pd.to_csv(file_path, index=False)
    print(' Writtent to  ', file_path)


def create_distribution(entity_ls, entity_type_ls):
    dic_ent = {}  # dictionary of dictionary
    for entity, entity_type in zip(entity_ls, entity_type_ls):
        if entity not in dic_ent:
            dic_type = {key: 0 for key in FILTERED_TYPES}
            dic_type[entity_type] = 1
            dic_ent[entity] = dic_type
        else:
            temp_dic_type = dic_ent[entity]
            temp_dic_type[entity_type] += 1
            dic_ent[entity] = temp_dic_type
    return dic_ent


def create_df_dic(lang_id_ls, dic_ent):
    pd_distro = pd.DataFrame(dic_ent)
    pd_distro = pd_distro.T
    pd_distro = add_label(pd_distro)
    file_name = str(lang_id_ls[0]) + '_' + 'entity_distro_' + '.csv'
    file_path = WRITE_PATH + file_name
    pd_distro.to_csv(file_path)


def main():
    for lang_id, sent_path, tag_path in zip(LANG_ID, SENT_FILE_PATH, TAG_FILE_PATH):
        print('Processing ', sent_path, 'with ', tag_path)
        sent_data = read_data(sent_path)
        tag_data = read_data(tag_path)
        save_to_pickle(sent_data, 'sent', lang_id)
        save_to_pickle(tag_data, 'label', lang_id)
        print('     Total len(sent_data) ', len(sent_data))
        print('     Total len(tag_data) ', len(tag_data))
        print('     TEST sent_data[0] ', sent_data[0])
        print('     TEST tag_data[0] ', tag_data[0])
        filtered_sent_id = []
        filtered_entity_ls = []
        filtered_entity_type_ls = []
        for k, v in tag_data.items():
            sent_id = k
            sent_label = v
            if is_single_word_entity(sent_label)[0]:
                entity_type = is_single_word_entity(sent_label)[1][0]
                index_0f_entity_type = sent_label.split().index(entity_type)
                entity = sent_data[sent_id].split()[index_0f_entity_type]
                filtered_sent_id.append(sent_id)
                filtered_entity_ls.append(entity)
                filtered_entity_type_ls.append(entity_type)
        print('     Total number of Filtered  ', len(filtered_sent_id))
        lang_id_ls = [lang_id] * len(filtered_sent_id)
        create_df_list(lang_id_ls, filtered_sent_id, filtered_entity_ls, filtered_entity_type_ls)
        dic_ent = create_distribution(filtered_entity_ls, filtered_entity_type_ls)
        create_df_dic(lang_id_ls, dic_ent)


if __name__ == "__main__":
    main()
