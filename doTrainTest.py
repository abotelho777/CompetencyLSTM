from __future__ import print_function

import code0_hyperParameter as code0
import code1_data as code1
import code2_model as code2
import code3_runEpoch as code3
import uril_tools as aux
import tensorflow as tf
import numpy as np
import pandas as pd

import datetime

np.set_printoptions(threshold=np.inf)


def run_model(run_type,include_model,cv = 1):
    from time import time
    start = time()

    timeStampe = datetime.datetime.now().strftime("%m-%d-%H-%M")
    result_file_name_train = 'resources/result_'+str(timeStampe)+'_'+str(run_type)+'_'+str(include_model)+'_train.csv'
    result_file_name_test = 'resources/result_'+str(timeStampe)+'_'+str(run_type)+'_'+str(include_model)+'_test.csv'
    result_file_name_est = 'resources/result_' + str(timeStampe) + '_' + str(run_type) + '_' + str(
        include_model) + '_test_estimates.csv'
    result_tmp = pd.DataFrame(columns=('rmse', 'auc', 'r2','cv','epoch'))
    result_tmp.to_csv(result_file_name_train)
    result_tmp.to_csv(result_file_name_test)

    code0.RUN_TYPE = run_type
    code0.INCLUDE_MODEL = include_model

    aux.check_directories()

    dt = code1.ReadData()
    dataset, labels = dt.create_label_and_delete_last_one()
    tuple_data = dt.convert_data_labels_to_tuples(dataset, labels)

    skill_num = max(dataset['skill_id'].unique()) + 1
    skill_set = dataset['skill_id'].unique()
    seq_width = len(dt.features_only_names)

    config = code0.ModelParamsConfig(skill_num, seq_width, skill_set)
    eval_config = code0.ModelParamsConfig(skill_num, seq_width, skill_set)

    if code0.NUM_STEPS != 0:
        config.num_steps = code0.NUM_STEPS
    else:
        config.num_steps = aux.get_num_step(dataset['user_id'])

    eval_config.num_steps = config.num_steps
    eval_config.batch_size = 2

    config.print_config('traing')
    eval_config.print_config('testing')

    # name_list = ['cv', 'epoch', 'type', 'rmse', 'auc', 'r2', 'inter_rmse', 'inter_auc', 'inter_r2', 'intra_rmse',
    #              'intra_auc', 'intra_r2']
    # result_data = pd.DataFrame(columns=name_list)
    if cv ==1:
        CVname = ['c1']
    elif cv ==5:
        CVname = ['c1', 'c2', 'c3', 'c4', 'c5']
    size = len(tuple_data)

    for index, cv_num_name in enumerate(CVname):
        # timeStampe = datetime.datetime.now().strftime("%m-%d-%H:%M")

        train_tuple_rows = tuple_data[:int(index * 0.2 * size)] + tuple_data[int((index + 1) * 0.2 * size):]
        test_tuple_rows = tuple_data[int(index * 0.2 * size): int((index + 1) * 0.2 * size)]

        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            # training model
            print("\n==> Load Training model")
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = code2.Model(is_training=True, config=config)

            # testing model
            print("\n==> Load Testing model")
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = code2.Model(is_training=False, config=eval_config)

            # tf.initialize_all_variables().run()
            tf.global_variables_initializer().run()

            print("==> begin to run epoch...")
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                rt = session.run(m.lr)
                train_result, estimates = code3.run_epoch(session, m, train_tuple_rows, m.train_op, verbose=True)

                # for e in range(10):
                #     print(estimates[e])

                print('Train ', cv_num_name, "-" * 10, str(i), '/',
                          str(config.max_max_epoch ), "-" * 10)
                print(train_result)

                train_result['cv'] = cv_num_name
                train_result['epoch'] = i
                train_result.to_csv(result_file_name_train, mode='a',header=False)

                display = 5
                if ((i + 1) % display == 0):
                    print ("\n")
                    print('Test ', cv_num_name, "=" * 30, str(int((i + 1) / display)), '/',
                          str(int((config.max_max_epoch + 1) / display)), "=" * 30)
                    test_result, estimates = code3.run_epoch(session, mtest, test_tuple_rows, tf.no_op())
                    test_result['cv'] = cv_num_name
                    test_result['epoch'] = int((i + 1) / display)
                    test_result.to_csv(result_file_name_test, mode='a',header=False)
                    print(test_result)
                    print('END--', "=" * 80)
                    # if int((i + 1) / display)>=8:
                    #     test_result['cv'] = cv_num_name
                    #     test_result['epoch'] = int((i + 1) / display)>=8
                    #     test_result.to_csv(result_file_name_test, mode='a',header=False)
                    #     print('++> save result to ', result_file_name_test)
    print("==> Finsih! whole process!")
    print("\n\n\nTotal Runtime: {:<.2f}s".format(time()-start))


def main(unused_args):
    for run_type in [1]:  #these file name can be found in the file: code1_data.py
        for include_model in [4]:
            try:
                run_model(run_type,include_model,cv=5)
            except Exception:
                timeStampe = datetime.datetime.now().strftime("%m-%d-%H-%M")
                result_file_name = 'resources/result_'+str(timeStampe)+'_'+str(run_type)+'_'+str(include_model)+'.csv'
                result = pd.DataFrame({'info':'ERROR'})
                result.to_csv(result_file_name)


if __name__ == "__main__":
    tf.app.run()
