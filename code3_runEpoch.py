from uril_tools import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import numpy as np
import pyprind
import code0_hyperParameter as code0

np.set_printoptions(threshold=np.inf)


def get_evaluate_result(actual_labels, pred_prob):
    rmse = sqrt(mean_squared_error(actual_labels, pred_prob))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    r2 = r2_score(actual_labels, pred_prob)
    return rmse, auc, r2


def run_epoch(session, m, students, eval_op, verbose=False):
    pred_prob = []
    actual_labels = []  # use for whole comparasion

    result = pd.DataFrame(columns=['rmse', 'auc', 'r2'])

    generNPC_whole_prob = []
    generNPC_whole_actual = []
    first_action_whole_prob = []
    first_action_whole_actual = []
    res_correct_whole_prob = []
    res_correct_whole_actual = []
    wheelspin_whole_prob = []
    wheelspin_whole_actual = []
    bored_whole_prob = []
    bored_whole_actual = []
    concentrating_whole_prob = []
    concentrating_whole_actual = []
    confused_whole_prob = []
    confused_whole_actual = []
    frustrated_whole_prob = []
    frustrated_whole_actual = []

    iteration = int(len(students) / m.batch_size)

    for i_iter in pyprind.prog_percent(range(iteration)):
        x = np.zeros((m.batch_size, m.num_steps, m.seq_width))

        target_id = np.array([], dtype=np.int32)
        target_correctness = []  # use for just a batch

        generNPC_idx = []
        generNPC_list = []
        first_action_idx = []
        first_action_list = []
        res_correct_idx = []
        res_correct_list = []
        wheelspin_idx = []
        wheelspin_list = []
        bored_idx = []
        bored_list = []
        concentrating_idx = []
        concentrating_list = []
        confused_idx = []
        confused_list = []
        frustrated_idx = []
        frustrated_list = []

        # load data for a batch
        # tuple formate
        # 0: user_id
        # 1: record_numb
        # 2: data
        # 3: Target_Id
        # 4: correctness
        for i_batch in range(m.batch_size):
            student = students[i_iter * m.batch_size + i_batch]

            record_num = student[code0.TUPLE_ID_DICT['record_numb']]
            record_content = student[code0.TUPLE_ID_DICT['data']].as_matrix()
            skill_id = student[code0.TUPLE_ID_DICT['Target_Id']]
            correctness = student[code0.TUPLE_ID_DICT['correctness']]

            generNPC_item = student[code0.TUPLE_ID_DICT['correctness']]
            first_action_item = student[code0.TUPLE_ID_DICT['first_action']]
            res_correct_item = student[code0.TUPLE_ID_DICT['res_correct']]
            wheelspin_item = student[code0.TUPLE_ID_DICT['wheelspin']]
            bored_item = student[code0.TUPLE_ID_DICT['bored']]
            concentrating_item = student[code0.TUPLE_ID_DICT['concentrating']]
            confused_item = student[code0.TUPLE_ID_DICT['confused']]
            frustrated_item = student[code0.TUPLE_ID_DICT['frustrated']]

            # construct data for training:
            # data ~ x
            # target_id ~ skill_id
            # target_correctness ~ correctness
            for i_recordNumb in range(record_num):
                if (i_recordNumb < m.num_steps):
                    x[i_batch, i_recordNumb, :] = record_content[i_recordNumb, :]

                    if skill_id[i_recordNumb] in m.skill_set:
                        temp = i_batch * m.num_steps * m.skill_num + i_recordNumb * m.skill_num + skill_id[i_recordNumb]
                    else:
                        temp = i_batch * m.num_steps + i_recordNumb * m.skill_num + 0

                    target_id = np.append(target_id, [[temp]])
                    target_correctness.append(int(correctness[i_recordNumb]))
                    actual_labels.append(int(correctness[i_recordNumb]))

                    if generNPC_item[i_recordNumb] != code0.LABEL_MASK:
                        generNPC_list.append(generNPC_item[i_recordNumb])
                        generNPC_idx.append(i_batch * m.num_steps + i_recordNumb)
                        generNPC_whole_actual.append(generNPC_item[i_recordNumb])

                    if first_action_item[i_recordNumb] != code0.LABEL_MASK:
                        first_action_list.append(first_action_item[i_recordNumb])
                        first_action_idx.append(i_batch * m.num_steps + i_recordNumb)
                        first_action_whole_actual.append(first_action_item[i_recordNumb])

                    if res_correct_item[i_recordNumb] != code0.LABEL_MASK:
                        res_correct_list.append(res_correct_item[i_recordNumb])
                        res_correct_idx.append(i_batch * m.num_steps + i_recordNumb)
                        res_correct_whole_actual.append(res_correct_item[i_recordNumb])

                    if wheelspin_item[i_recordNumb] != code0.LABEL_MASK:
                        wheelspin_list.append(wheelspin_item[i_recordNumb])
                        wheelspin_idx.append(i_batch * m.num_steps + i_recordNumb)
                        wheelspin_whole_actual.append(wheelspin_item[i_recordNumb])

                    if bored_item[i_recordNumb] != code0.LABEL_MASK:
                        bored_list.append(bored_item[i_recordNumb])
                        bored_idx.append(i_batch * m.num_steps + i_recordNumb)
                        bored_whole_actual.append(bored_item[i_recordNumb])

                    if concentrating_item[i_recordNumb] != code0.LABEL_MASK:
                        concentrating_list.append(concentrating_item[i_recordNumb])
                        concentrating_idx.append(i_batch * m.num_steps + i_recordNumb)
                        concentrating_whole_actual.append(concentrating_item[i_recordNumb])

                    if confused_item[i_recordNumb] != code0.LABEL_MASK:
                        confused_list.append(confused_item[i_recordNumb])
                        confused_idx.append(i_batch * m.num_steps + i_recordNumb)
                        confused_whole_actual.append(confused_item[i_recordNumb])

                    if frustrated_item[i_recordNumb] != code0.LABEL_MASK:
                        frustrated_list.append(frustrated_item[i_recordNumb])
                        frustrated_idx.append(i_batch * m.num_steps + i_recordNumb)
                        frustrated_whole_actual.append(frustrated_item[i_recordNumb])
                else:
                    break

        pred, first_action_pred, res_correct_pred, wheelspin_pred, bored_pred, concentrating_pred, confused_pred, frustrated_pred, generNPC_pred, _ = session.run(
            [m.pred, m.first_action_pred, m.res_correct_pred, m.wheelspin_pred, m.bored_pred, m.concentrating_pred,
             m.confused_pred, m.frustrated_pred, m.generNPC_pred, eval_op],
            feed_dict={m.inputs: x, m.target_id: target_id, m.target_correctness: target_correctness,
                       m.first_action_idx: first_action_idx, m.first_action_list: first_action_list,
                       m.res_correct_idx: res_correct_idx, m.res_correct_list: res_correct_list,
                       m.wheelspin_idx: wheelspin_idx, m.wheelspin_list: wheelspin_list, m.bored_idx: bored_idx,
                       m.bored_list: bored_list, m.concentrating_idx: concentrating_idx,
                       m.concentrating_list: concentrating_list, m.confused_idx: confused_idx,
                       m.confused_list: confused_list, m.frustrated_idx: frustrated_idx,
                       m.frustrated_list: frustrated_list, m.generNPC_idx: generNPC_idx,
                       m.generNPC_list: generNPC_list})

        estimates = np.array([np.array(pred), np.array(first_action_pred), np.array(res_correct_pred),
                              np.array(wheelspin_pred), np.array(bored_pred), np.array(concentrating_pred),
                              np.array(confused_pred), np.array(frustrated_pred), np.array(generNPC_pred)]).T

        for p in pred:
            pred_prob.append(p)
        for f in first_action_pred:
            first_action_whole_prob.append(f)
        for f in res_correct_pred:
            res_correct_whole_prob.append(f)
        for f in wheelspin_pred:
            wheelspin_whole_prob.append(f)
        for f in bored_pred:
            bored_whole_prob.append(f)
        for f in concentrating_pred:
            concentrating_whole_prob.append(f)
        for f in confused_pred:
            confused_whole_prob.append(f)
        for f in frustrated_pred:
            frustrated_whole_prob.append(f)
        for k in generNPC_pred:
            generNPC_whole_prob.append(k)

    if code0.INCLUDE_MODEL == 1:
        result.loc['npc', :] = get_evaluate_result(actual_labels, pred_prob)
    elif code0.INCLUDE_MODEL == 2:
        result.loc['wheelspin', :] = get_evaluate_result(wheelspin_whole_actual, wheelspin_whole_prob)
    elif code0.INCLUDE_MODEL == 3:
        result.loc['first_action', :] = get_evaluate_result(first_action_whole_actual, first_action_whole_prob)
        result.loc['wheelspin', :] = get_evaluate_result(wheelspin_whole_actual, wheelspin_whole_prob)
    elif code0.INCLUDE_MODEL == 4:
        result.loc['npc', :] = get_evaluate_result(actual_labels, pred_prob)
        result.loc['first_action', :] = get_evaluate_result(first_action_whole_actual, first_action_whole_prob)
        result.loc['res_correct', :] = get_evaluate_result(res_correct_whole_actual, res_correct_whole_prob)
        result.loc['wheelspin', :] = get_evaluate_result(wheelspin_whole_actual, wheelspin_whole_prob)
        result.loc['bored', :] = get_evaluate_result(bored_whole_actual, bored_whole_prob)
        result.loc['concentrating', :] = get_evaluate_result(concentrating_whole_actual, concentrating_whole_prob)
        result.loc['confused', :] = get_evaluate_result(confused_whole_actual, confused_whole_prob)
        result.loc['frustrated', :] = get_evaluate_result(frustrated_whole_actual, frustrated_whole_prob)
    elif code0.INCLUDE_MODEL == 5:
        result.loc['res_correct', :] = get_evaluate_result(res_correct_whole_actual, res_correct_whole_prob)
    elif code0.INCLUDE_MODEL == 6:
        result.loc['generNPC', :] = get_evaluate_result(generNPC_whole_actual, generNPC_whole_prob)

    return result, estimates


if __name__ == "__main__":
    pass
