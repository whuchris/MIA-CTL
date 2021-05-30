import os
import random

def generate_datatxt(data_dir = r"./dataset/"):
    '''
    Read from the dataset and make file directory
    :param data_dir: the whole dataset directory
    :return: result in a 'data.txt' file
    '''
    png_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file:
                png_files.append(os.path.join(root,file))
        break
    with open(r"./data_folder/data.txt",'w') as f:
        for file in png_files:
            # modify the following code if neccesary
            # Our origin dataset file format are like:
            # ./dataset/0_info1_info2_patientname_info3.png
            label = file.split('/')[-1].split('_')[0]
            name = file.split('_')[3]
            f.write(file+'\t'+label+'\t'+name+'\n')

def split_ssl_and_sl(ssl_rate = 0.2,whole_file=r"./data_folder/data.txt"):

    '''
    Divide the whole datasets into self-supervised sub-dataset and supervised sub-dataset.

    :param ssl_rate: the rate of self-supervised sub-dataset.
    :param whole_file: directory of whole dataset text file.
    :return: no things returned. Result in two text files.

    '''

    dict = {'0':0,'1':1,'2':2,'3':3,'4':4}
    lists = [[] for i in range(5)]
    patient_lists = [[] for i in range(5)]
    with open(whole_file,'r') as file:
        line = file.readline()
        while line:
            file_name = line.split('\t')[0]
            tmp_class_name = line.split('\t')[1]
            tmp_patient_id = line.split('\t')[2].split('\n')[0]
            # print(file_name)
            # print(tmp_patient_id+'\t'+tmp_class_name)
            lists[dict[tmp_class_name]].append(file_name)
            if tmp_patient_id not in patient_lists[dict[tmp_class_name]]:
                patient_lists[dict[tmp_class_name]].append(tmp_patient_id)
            line = file.readline()
    file.close()
    print(len(patient_lists[0]))
    print(len(patient_lists[1]))
    print(len(patient_lists[2]))
    print(len(patient_lists[3]))
    print(len(patient_lists[4]))

    for i in range(5):
        random.shuffle(patient_lists[i])

    self_supervised_patient_lists = [[] for i in range(5)]
    supervised_patient_lists = [[] for i in range(5)]
    for i in range(5):
        tmp_ssl = patient_lists[i][0:int(ssl_rate*len(patient_lists[i]))]
        tmp_sl = patient_lists[i][int(ssl_rate*len(patient_lists[i])):]
        for train_item in tmp_ssl:
            self_supervised_patient_lists[i].append(train_item)
        for test_item in tmp_sl:
            supervised_patient_lists[i].append(test_item)
    # print(len(train_patient_lists),len(test_patient_lists))
    # print(train_patient_lists)
    # print(test_patient_lists)

    ssl_list_file = open(r"./data_folder/self_supervised_list_folder.txt",'w')
    sl_list_file = open(r"./data_folder/supervised_folder.txt",'w')

    with open(whole_file,'r') as file:
        line = file.readline()
        while line:
            file_name = line.split('\t')[0]
            tmp_class_name = line.split('\t')[1]
            tmp_patient_id = line.split('\t')[2].split('\n')[0]
            tmp_class = dict[tmp_class_name]
            if tmp_patient_id in self_supervised_patient_lists[tmp_class]:
                ssl_list_file.write(file_name+'\t'+tmp_class_name+'\t'+ tmp_patient_id+'\n')
            elif tmp_patient_id in supervised_patient_lists[tmp_class]:
                sl_list_file.write(file_name+'\t'+tmp_class_name+'\t'+ tmp_patient_id+'\n')
            line = file.readline()
    file.close()
    ssl_list_file.close()
    ssl_list_file.close()

def generate_folder(folder_num = 10,whole_file=r"./data_folder/supervised_folder.txt"):

    '''
    Generate 10 folders used to perform 10-fold cross validation experiment.

    :param folder_num: num of folder(can be 3,5 or 10)
    :param whole_file: directory of whole dataset text file.
    :return: resulting in 10 [train/val]_folder_i.txts.
    '''

    dict = {'0':0,'1':1,'2':2,'3':3,'4':4}
    lists = [[] for i in range(5)]
    patient_lists = [[] for i in range(5)]
    with open(whole_file,'r') as file:
        line = file.readline()
        while line:
            file_name = line.split('\t')[0]
            tmp_class_name = line.split('\t')[1]
            tmp_patient_id = line.split('\t')[2].split('\n')[0]
            # print(file_name)
            # print(tmp_patient_id+'\t'+tmp_class_name)
            lists[dict[tmp_class_name]].append(file_name)
            if tmp_patient_id not in patient_lists[dict[tmp_class_name]]:
                patient_lists[dict[tmp_class_name]].append(tmp_patient_id)
            line = file.readline()
    file.close()
    # print(len(patient_lists[0]))
    # print(len(patient_lists[1]))
    # print(len(patient_lists[2]))
    # print(len(patient_lists[3]))
    # print(len(patient_lists[4]))
    for i in range(5):
        random.shuffle(patient_lists[i])

    for folder in range(folder_num):
        train_patient_lists = [[] for i in range(5)]
        test_patient_lists = [[] for i in range(5)]
        for i in range(5):
            tmp_train = patient_lists[i][:int(folder/folder_num * len(patient_lists[i]))]+patient_lists[i][int((folder+1)/folder_num * len(patient_lists[i])):]
            tmp_test = patient_lists[i][int(folder/folder_num * len(patient_lists[i])):int((folder+1)/folder_num * len(patient_lists[i]))]
            for train_item in tmp_train:
                train_patient_lists[i].append(train_item)
            for test_item in tmp_test:
                test_patient_lists[i].append(test_item)
        # print(len(train_patient_lists),len(test_patient_lists))
        # print(train_patient_lists)
        # print(test_patient_lists)

        train_list_file = open(r"./data_folder/train_folder_"+str(folder)+".txt",'w')
        test_list_file = open(r"./data_folder/val_folder_"+str(folder)+".txt",'w')

        with open(whole_file,'r') as file:
            line = file.readline()
            while line:
                file_name = line.split('\t')[0]
                tmp_class_name = line.split('\t')[1]
                tmp_patient_id = line.split('\t')[2].split('\n')[0]
                tmp_class = dict[tmp_class_name]
                if tmp_patient_id in train_patient_lists[tmp_class]:
                    train_list_file.write(file_name+'\t'+tmp_class_name+'\n')
                elif tmp_patient_id in test_patient_lists[tmp_class]:
                    test_list_file.write(file_name+'\t'+tmp_class_name+'\n')
                line = file.readline()
        file.close()

if __name__ == '__main__':
    # generate_datatxt()
    # split_ssl_and_sl()
    # generate_folder()
    pass

