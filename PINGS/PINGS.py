from re import T
from scipy.optimize.minpack import _initialize_feasible
from utils import *
import time
from matplotlib import pyplot as plt
import numpy as np




missing_rate=0.15
nodes_num=750
incomplete_choice = 1
small_times=4
big_times=6
sample_choice = 0
equation_flag_s = False
equation_flag_lambda = False
stop_threshold = 10
initial_epsilon = 0.001
construct_choice = 0
degree =4
epsilon_change=False



graph_path='./example_network.txt'
result_path='./example_record_data.txt'
print("graph_path=%s, result_path=%s"%(graph_path, result_path,))


print("missing_rate=%f, nodes_num=%d, incomplete_choice=%d, samll_times=%d, big_times=%d, "
      "sample_choice=%d, equation_flag_s=%r, equation_flag_lambda=%r, stop_threshold=%f, initial_epsilon=%f, construct_choice=%d, epsilon_change=%r"%(missing_rate, nodes_num,
                                                                           incomplete_choice, small_times,
                                                                           big_times,sample_choice,equation_flag_s,equation_flag_lambda,
                                                                           stop_threshold, initial_epsilon,construct_choice, epsilon_change))


ground_truth_network, diffusion_result = load_data(graph_path, result_path)
p = ground_truth_network*0.3


incomplete_result = make_incomplete(diffusion_result, missing_rate, nodes_num, incomplete_choice, missing_index_address)

results_list, prior_prob, temp_results_list= init_sample(incomplete_result, small_times, big_times, sample_choice)
init_sample_accuracy=cal_sample_accuracy(temp_results_list, incomplete_result, diffusion_result, small_times, big_times)


one_cnt_list = np.zeros(big_times)
for i in range(big_times):
    result_i = results_list[i]
    one_cnt = np.sum(result_i[np.where(incomplete_result==-1)])
    one_cnt_list[i]=one_cnt

it_cnt = 1
pre_network_stru = np.zeros(ground_truth_network.shape)

begin_all = time.time()
while True:
    it_begin = time.time()

    if construct_choice==0:
        network_list = construct_network_icde(results_list, big_times)
    elif construct_choice==1:
        network_list = construct_network_twind(results_list, big_times)

    for i in range(big_times):
        precision, recall, f_score = cal_F1(ground_truth_network, network_list[i])
        print(precision)
        print(recall)
        print(f_score)
        print("--------------------")

    comb_network = combine_network(network_list)
    change_num = np.sum(abs(comb_network - pre_network_stru))
    if change_num==0:
        break

    pre_network_stru = comb_network.copy()
    grd_begin = time.time()
    grd_it_cnt = 1
    while True:
        inner_begin = time.time()
        s_matrix = cal_s(p_matrix, incomplete_result, comb_network, prior_prob, equation_flag_s)

        lambda_matrix = cal_lambda(p_matrix, s_matrix, comb_network, incomplete_result, equation_flag_lambda)
        pre_p_matrix = p_matrix.copy()

        p_matrix = update_p(p_matrix, s_matrix, lambda_matrix, comb_network, incomplete_result, initial_epsilon, grd_it_cnt, epsilon_change)

        grd_it_cnt += 1
        delta_p_matrix = np.sum(abs(p_matrix - pre_p_matrix))
        if delta_p_matrix<stop_threshold:
            break
    grd_end = time.time()

    s_matrix = cal_s(p_matrix, incomplete_result, comb_network, prior_prob, equation_flag_s)

    results_list, temp_results_list= sample_data_s(s_matrix, incomplete_result, small_times, big_times)
    sample_accuracy=cal_sample_accuracy(temp_results_list, incomplete_result, diffusion_result, small_times, big_times

    one_cnt_list = np.zeros(big_times)
    for i in range(big_times):
        result_i = results_list[i]
        one_cnt = np.sum(result_i[np.where(incomplete_result == -1)])
        one_cnt_list[i] = one_cnt

    it_end = time.time()

    if it_cnt>5:
        break

    it_cnt += 1
