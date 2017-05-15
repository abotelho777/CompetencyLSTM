NUM_STEPS = 0

DEBUG_PRINT = True

"""
1. NPC
2. wheel_spin
# 3. wheel_spin,first_action
4. all
5. retention
6. generNPC
"""
INCLUDE_MODEL = 2   #run model

"""
1. compress 92 features
2. compress cross featuress
3. compress 92 features and cross features
"""
RUN_TYPE = 2    #load file name



DATASETSIZE = "large"  # 'large | small'
RNN_layer_number = 1  # '1|2'
CELLTYPE = "LSTM"  # "RNN | LSTM | GRU"

TUPLE_ID_DICT ={
    'user_id':0,
    'record_numb':1,
    'data':2,
    'Target_Id':3,
    'correctness':4,
    'first_action':5,
    'res_correct':6,
    'wheelspin':7,
    'bored':8,
    'concentrating':9,
    'confused':10,
    'frustrated':11
}

LABEL_MASK = -1
DATA_MASK = 0
CORRECT_NAME = 'correct'



class ModelParamsConfig(object):
    def __init__(self, skill_num,seq_width,skill_set):
        self.num_steps = 0  # need to resign value of time stampes
        self.skill_num = skill_num  # need to resign value of skill number
        self.seq_width = seq_width  # need to resign value
        self.skill_set = skill_set

        self.batch_size = 10

        self.max_max_epoch = 40
        self.num_layer = RNN_layer_number
        self.cell_type = CELLTYPE  # "RNN | LSTM | GRU"
        self.hidden_size = 200
        self.hidden_size_2 = 200

        self.init_scale = 0.05
        self.learning_rate = 0.05
        self.max_grad_norm = 4
        self.max_epoch = 5
        self.keep_prob = 0.6

        self.lr_decay = 0.9
        self.momentum = 0.95
        self.min_lr = 0.0001


    def print_config(self,name):
        print ('\n')
        print ('==> ',name, "\tconfiguration")
        print ('++> num_steps\t',self.num_steps)
        print ('++> skill_num\t',self.skill_num)
        print ('++> seq_width\t',self.seq_width)
        print ('++> batch_size\t',self.batch_size)
        print ('++> hidden_size\t',self.hidden_size)
