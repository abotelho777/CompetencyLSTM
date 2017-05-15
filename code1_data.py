import pandas as pd
import uril_tools as aux
import os, sys
import pyprind as pp
import code0_hyperParameter as code0
import random
import numpy as np


class ReadData(object):
    def __init__(self):

        if code0.RUN_TYPE == 1:
            file_name = 'resources/labeled_compressed92features.csv'
        elif code0.RUN_TYPE == 2:
            file_name = './data/labeled_compressedcrossfeatures.csv'
        elif code0.RUN_TYPE == 3:
            file_name = './data/labeled_combinedfeatures.csv'
        else:
            pass

        # if code0.RUN_TYPE == 1:
        #     file_name = './data/3_stacked/half_with_Dropout/labeled_compressed92features.csv'
        # elif code0.RUN_TYPE == 2:
        #     file_name = './data/3_stacked/half_with_Dropout/labeled_compressedcrossfeatures.csv'
        # elif code0.RUN_TYPE == 3:
        #     file_name = './data/3_stacked/half_with_Dropout/labeled_combinedfeatures.csv'


        # if code0.RUN_TYPE == 4:
        #     file_name = './data/4_edm_after/labeled_92features.csv'
        # elif code0.RUN_TYPE == 5:
        #     file_name = './data/4_edm_after/labeled_compressed92features.csv'
        # elif code0.RUN_TYPE == 6:
        #     file_name = './data/3_stacked/half_stacked_with_Dropout/labeled_combinedfeatures.csv'



        # if code0.RUN_TYPE == 4:
        #     file_name = './data/3_stacked/half_stacked_with_Dropout/labeled_compressed92features.csv'
        # elif code0.RUN_TYPE == 5:
        #     file_name = './data/3_stacked/half_stacked_with_Dropout/labeled_compressedcrossfeatures.csv'
        # elif code0.RUN_TYPE == 6:
        #     file_name = './data/3_stacked/half_stacked_with_Dropout/labeled_combinedfeatures.csv'

        # elif code0.RUN_TYPE == 7:
        #     file_name = './data/3_stacked/half_Affect_stacked_CF_with_Dropout/labeled_compressed92features.csv'
        # elif code0.RUN_TYPE == 8:
        #     file_name = './data/3_stacked/half_Affect_stacked_CF_with_Dropout/labeled_compressedcrossfeatures.csv'
        # elif code0.RUN_TYPE == 9:
        #     file_name = './data/3_stacked/half_Affect_stacked_CF_with_Dropout/labeled_combinedfeatures.csv'
        #     pass

        # if code0.RUN_TYPE == 1:
        #     file_name = '/home/leon/projects/joe-tensorflow/data/PCA/pca_data.csv'

        self.file_name = file_name
        self.process_file_name = os.path.dirname(self.file_name) + '/processed_' + str(code0.RUN_TYPE) + '_' + \
                                 os.path.split(file_name)[-1]

        if os.path.exists(self.process_file_name):
            self.data = pd.read_csv(self.process_file_name)
            print('==> load processed file ', self.process_file_name, ' directly')
        else:
            self.data = self.__get_csv_raw_data(file_name)

        self.data_feature_names = ['user_id', 'skill_id']
        self.features_only_names = []
        for item in self.data.columns:
            if item.find('feature') != -1:
                self.data_feature_names.append(item)
                self.features_only_names.append(item)

    def __convert_NPC(self, data):
        print('==> convert NPC to 0,1 or ' + str(code0.LABEL_MASK))
        npc_list = data[code0.CORRECT_NAME]
        new_npc_list = []
        for idx in pp.prog_percent(range(len(npc_list)), stream=sys.stdout):
            if npc_list[idx] >= 0 and npc_list[idx] < 0.5:
                new_npc_list.append(0)
            elif npc_list[idx] >= 0.5 and npc_list[idx] <= 1:
                new_npc_list.append(1)
            else:
                new_npc_list.append(code0.LABEL_MASK)
        data[code0.CORRECT_NAME] = new_npc_list
        return data

    # def __shift_label(self, data, feature_name):
    #     """
    #     shift label since it lable current exercise
    #     feature name:
    #     :return:
    #     """
    #     print('==> shift ', feature_name)
    #     user_id_list = list(data['user_id'])
    #     item_id_list = list(data[feature_name])
    #     new_item_id_list = []
    #
    #     for idx in pp.prog_percent(range(len(user_id_list) - 1), stream=sys.stdout):
    #         if user_id_list[idx] == user_id_list[idx + 1]:
    #             new_item_id_list.append(item_id_list[idx + 1])
    #         else:
    #             new_item_id_list.append(code0.LABEL_MASK)
    #     new_item_id_list.append(code0.LABEL_MASK)
    #     assert len(new_item_id_list) == len(item_id_list)
    #
    #     data[feature_name] = new_item_id_list
    #     return data

    def __get_csv_raw_data(self, file_name):
        print('==> read raw csv data')
        data = pd.read_csv(file_name)
        print('==> columns names\t', data.columns)
        data = data.fillna(code0.LABEL_MASK)
        data = self.__convert_NPC(data)

        # if action_shift:
        #     data = self.__shift_label(data, 'first_action')

        print("==> BEGIN convert skill_id")
        temp_set = sorted(set(list(data['skill_id'])))
        temp_dict = {key: value + 1 for value, key in enumerate(temp_set)}
        data['skill_id'].replace(temp_dict, inplace=True)
        print("==> END  convert skill_id")

        data.to_csv(self.process_file_name, index=False)
        print('==> save processed file to ', self.process_file_name)
        return data

    def create_label_and_delete_last_one(self):
        userID_Quest_number_matrix = aux.getUserQuesNumList(self.data['user_id'])  # user_id: number of questions
        print("==> creat skill_id+label, last record of every user is deleted")
        print("==> delete user whose problem number is less than 2")
        print("==> shift correctness and first action label")
        row_size = len(self.data);
        index = 0
        kindex = 0
        dataset = pd.DataFrame()
        labels = pd.DataFrame()

        bar = pp.ProgPercent(row_size, stream=sys.stdout)
        while (index < row_size):
            id_number = userID_Quest_number_matrix[kindex, 1]
            if id_number > 2:
                dataTemp = self.data.loc[index:index + id_number - 2, self.data_feature_names]
                labeTemp = pd.DataFrame({'user_id': int(self.data.loc[index, 'user_id']), 'label_skill_id': list(
                    self.data.loc[index + 1:index + id_number - 1, "skill_id"]), 'label_correct': list(
                    self.data.loc[index + 1:index + id_number - 1, 'correct']), 'first_action': list(
                    self.data.loc[index + 1:index + id_number - 1, 'first_action']),
                                         'res_correct': list(self.data.loc[index:index + id_number - 2, 'res_correct']),
                                         'wheelspin': list(self.data.loc[index:index + id_number - 2, 'wheelspin']),
                                         'bored': list(self.data.loc[index:index + id_number - 2, 'bored']),
                                         'concentrating': list(
                                             self.data.loc[index:index + id_number - 2, 'concentrating']),
                                         'confused': list(self.data.loc[index:index + id_number - 2, 'confused']),
                                         'frustrated': list(self.data.loc[index:index + id_number - 2, 'frustrated'])})
                if len(dataTemp) != len(labeTemp):
                    print(self.data.loc[index:index + id_number - 1])
                    print('--' * 30)
                    print(len(dataTemp))
                    print('--' * 30)
                    print(len(labeTemp))
                    print('--' * 30)
                    print(dataTemp)
                    print('--' * 30)
                    print(labeTemp)
                    assert len(dataTemp) == len(labeTemp)

                dataset = dataset.append(dataTemp)
                labels = labels.append(labeTemp)
                del dataTemp, labeTemp
            bar.update(id_number)
            index += id_number
            kindex += 1
        dataset = dataset.reset_index(drop=True)
        labels = labels.reset_index(drop=True)
        assert len(dataset) == len(labels), "dateset size\t" + str(len(dataset)) + "\tlabels size\t" + str(len(labels))
        self.PRINT_DATA_INFO(labels)
        return dataset, labels

    def convert_data_labels_to_tuples(self, dataset, labels):
        index = 0
        kindex = 0
        tuple_rows = []
        userID_Quest_number_matrix = aux.getUserQuesNumList(dataset['user_id'])
        print("==> convert data and labels to tuples")
        # tuple formate
        # 0: user_id
        # 1: record_numb
        # 2: data
        # 3: Target_Id
        # 4: correctness
        # 5: first_action
        # 6: res_correct
        # 7: wheelspin
        # 8: bored
        # 9: concentrating
        # 10: confused
        # 11: frustrated
        dataset_size = len(self.data)
        bar = pp.ProgPercent(dataset_size, stream=sys.stdout)

        while index < dataset_size:
            numb = int(userID_Quest_number_matrix[kindex, 1])
            assert int(userID_Quest_number_matrix[kindex, 0]) == int(dataset.loc[index, "user_id"])
            tup = (dataset.loc[index, "user_id"], numb, dataset.loc[index:index + numb - 1, self.features_only_names],
                   list(labels.loc[index:index + numb - 1, "label_skill_id"]),
                   # the input is a list but not pd.DataFrame, don't need to reset the index.
                   list(labels.loc[index:index + numb - 1, "label_correct"]),
                   list(labels.loc[index:index + numb - 1, "first_action"]),
                   list(labels.loc[index:index + numb - 1, "res_correct"]),
                   list(labels.loc[index:index + numb - 1, "wheelspin"]),
                   list(labels.loc[index:index + numb - 1, "bored"]),
                   list(labels.loc[index:index + numb - 1, "concentrating"]),
                   list(labels.loc[index:index + numb - 1, "confused"]),
                   list(labels.loc[index:index + numb - 1, "frustrated"]))
            # pd.DataFrame, loc and iloc cut differentsize!
            tuple_rows.append(tup)
            index += numb
            kindex += 1
            if kindex >= np.shape(userID_Quest_number_matrix)[0]:
                break
            bar.update(numb)
        random.shuffle(tuple_rows)
        return tuple_rows

    def PRINT_DATA_INFO(self, labels):
        if code0.DEBUG_PRINT:
            print('++> user number\t', len(aux.getUserQuesNumList(labels['user_id'])))
            print('++> record number\t', len(labels))
            for item in ['label_correct', 'first_action', 'res_correct', 'wheelspin', 'bored', 'concentrating',
                         'confused', 'frustrated']:
                print('++> ', item, "\t", aux.counter(labels[item]))


def check_data():
    file_name_old = './data/labeled_combinedfeatures.csv'
    file_name_new = './data/processed_3_labeled_combinedfeatures.csv'
    file_old = pd.read_csv(file_name_old)
    file_new = pd.read_csv(file_name_new)

    print(file_old.columns)
    print(file_new.columns)

    # print (file_old.head(5))
    # print (file_new.head(5))
    #
    # file_old.head(5).to_csv('./data/k_old.csv')
    # file_new.head(5).to_csv('./data/k_new.csv')


def read_pca_data():
    origin_name = '/home/leon/projects/joe-tensorflow/data/PCA/norm_affect_features.csv'
    pca_name = '/home/leon/projects/joe-tensorflow/data/PCA/norm_affect_features_PCA.csv'
    file_name = './data/3_stacked/half_with_Dropout/labeled_compressed92features.csv'

    orgin_data = pd.read_csv(origin_name)
    pca_data = pd.read_csv(pca_name)
    label_data = pd.read_csv(file_name)

    print(orgin_data.columns)
    print(pca_data.columns)
    print(label_data.columns)

    print(np.shape(orgin_data))
    print(np.shape(pca_data))
    print(np.shape(label_data))

    labels = label_data[['skill_id', 'user_id', 'assignment_id', 'problem_log_id', 'correct', 'first_action',
                        'res_delay',
                'res_correct', 'wheelspin', 'bored', 'concentrating', 'confused', 'frustrated']]

    columns_dict ={}
    for idx,item in enumerate(pca_data.columns):
        columns_dict[item] = 'featurs_'+str(idx)
    print (columns_dict)
    pca_data = pca_data.rename(columns=columns_dict)
    print(pca_data.columns)

    data = pd.concat([labels, pca_data], axis=1)
    print(data.columns)
    print(np.shape(data))
    data.to_csv('/home/leon/projects/joe-tensorflow/data/PCA/pca_data.csv')


if __name__ == '__main__':
    # dt = ReadData()
    # dataset, labels = dt.create_label_and_delete_last_one()
    # tuple_data = dt.convert_data_labels_to_tuples(dataset, labels)
    read_pca_data()
