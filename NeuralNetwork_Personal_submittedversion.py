import numpy as np
import random
import pandas as pd
import pickle
import matplotlib.pyplot as mlt
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    lambda_val = random.uniform(0, 1)
    learning_rate = random.uniform(0,
                                   0.5)  # how much of the change to absorb, (value comes from matlab or use random start and decay it accordingly)
    layers = ['input', 'hidden', 'output']  # 3 layers
    neurons = []  # amount of neurons per layer: default (2 inp, 2 out) -> hidden neuron can be any number
    weights = []
    input_value = [0.0, 0.0]
    delta = []

    def set_neurons(self, neurons):
        temp_list = list()
        for neu in neurons:
            for i in range(neu):
                temp_list.append(0)

        self.neurons.append(temp_list[0:2])
        self.neurons.append(temp_list[2:len(temp_list) - 2])
        self.neurons.append(temp_list[len(temp_list) - 2:len(temp_list)])

    def weight_multiplier(self, input_value, weights):
        self.neurons[0] = input_value
        list_v = list()
        # print(len(self.weights))
        start = 0
        for i in range(len(weights)):
            sum = 0
            for j in range(len(input_value)):
                # print("jkasjf", input_value[j], weights[i][j])
                sum += input_value[j] * weights[i][j]
            list_v.append(sum)
            # print(sum)

        # print("Here>>>:")
        # print(list_v)
        # print(self.neurons)
        return list_v

    def __init__(self, neurons):
        self.v2 = [None]
        self.v1 = [None]
        self.set_neurons(neurons)

    def check(self):
        return self.neurons, "->", self.layers

    def activation_function(self, input_value, weights):
        v_list = self.weight_multiplier(input_value, weights)
        # print(val_1, val_2)
        list_actfun = list()
        for i in range(len(v_list)):
            list_actfun.append((1 / (1 + np.exp((-self.lambda_val * v_list[i])))))

        # print(list_actfun)
        self.neurons[1] = list_actfun
        # print(self.neurons)
        return list_actfun

    def error_calc(self, actual, predicted):
        return actual[0] - predicted[0], actual[1] - predicted[1]

    def backprop(self):
        pass

    def decay_lr(self):
        decay_rate = 0.5
        self.learning_rate = self.learning_rate * decay_rate

    def regularizer_adjuster(self, error):
        print("current lambda: ", self.lambda_val)
        if error[0][0] < error[1][0] or error[0][1] < error[1][1]:
            self.lambda_val = self.lambda_val * 0.85
            self.lambda_val = round(self.lambda_val, 4)
        else:
            self.lambda_val = self.lambda_val * 1.15
            self.lambda_val = round(self.lambda_val, 4)

        print("new lambda: ", self.lambda_val)

    def outgradient(self, output, error):
        o_grad = list()
        for i in range(len(output)):
            # print(output[i])
            # print(error[i])
            o_grad.append((self.lambda_val * output[i]) * (1 - output[i]) * error[i])
        # print(o_grad)
        return o_grad

    def local_gradient(self, hidden_val, out_gradient, hidden_weights):

        local_gradient = list()
        temp = list()
        sum = 0
        for i in range(len(hidden_weights)):
            for j in range(len(hidden_weights[i])):
                sum = sum + hidden_weights[j][i] * out_gradient[i]

            temp.append(sum)
        for i in range(len(out_gradient)):
            local_gradient.append(((self.lambda_val * hidden_val[i]) * (1 - hidden_val[i])) * (temp[i]))

        return local_gradient

    def weight_updation(self, out_gradient, hidden_val, hidden_weights):
        delta_w = list()
        for i in range(len(out_gradient)):
            for j in range(len(hidden_val)):
                delta_w.append(self.learning_rate * out_gradient[i] * hidden_val[j])
        # print(delta_w)
        temp = 0
        updated_weights_hidden = list()
        for i in range(len(hidden_weights)):
            for j in range(len(hidden_weights[i])):
                updated_weights_hidden.append(hidden_weights[j][i] + delta_w[temp])
                temp += 1

        return updated_weights_hidden


def RMSE(error):
    import math
    rmse = 0
    sum_1 = 0
    sum_2 = 0
    for i in range(len(error)):
        sum_1 += math.pow(error[i][0], 2)
        sum_2 += math.pow(error[i][1], 2)

    return math.sqrt(sum_1 / len(error)), math.sqrt(sum_2 / len(error))


def save_weights(weights, howeights):
    with open('last_weights_i_h.pkl', 'wb') as f:
        pickle.dump(weights, f)
    with open('last_weights_h_o.pkl', 'wb') as f:
        pickle.dump(howeights, f)


def load_weights(str_arg="best"):
    if str_arg == "last":
        with open('last_weights_i_h.pkl', 'rb') as f:
            in_hid_weights = pickle.load(f)
        with open('last_weights_h_o.pkl', 'rb') as f:
            hid_out_weights = pickle.load(f)
    else:
        with open('best_weights_i_h.pkl', 'rb') as f:
            in_hid_weights = pickle.load(f)
        with open('best_weights_h_o.pkl', 'rb') as f:
            hid_out_weights = pickle.load(f)

    return in_hid_weights, hid_out_weights


def inference(data, d_i_weights, d_h_weights):
    Neural_Network = NeuralNetwork([2, 2, 2])
    # add while input is coming, read input ~
    input_value = [data['x_axis_input'][0], data['y_axis_input'][0]]
    # print("Here:", d_i_weights)
    # print("Here:", d_h_weights)
    # print(d_i_weights[0],d_i_weights[1],d_i_weights[2],d_i_weights[3])
    weights = [[d_i_weights[0][0], d_i_weights[0][1]], [d_i_weights[1][0], d_i_weights[1][
        1]]]  # [[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]]
    # print(d_h_weights[0],d_h_weights[1],d_h_weights[2],d_h_weights[3])
    howeights = [[d_h_weights[0][0], d_h_weights[0][1]], [d_h_weights[1][0], d_h_weights[1][
        1]]]  # [[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]]
    hidden = Neural_Network.activation_function(input_value, weights)
    output = Neural_Network.activation_function(hidden, howeights)

    return output


def test_network(data, d_i_weights, d_h_weights):
    Neural_Network = NeuralNetwork([2, 2, 2])
    list_error = list()
    dict_1 = dict()
    for i in range(len(data)):
        input_value = [data['x_axis_input'][i], data['y_axis_input'][i]]
        weights = [[d_i_weights[0], d_i_weights[1]], [d_i_weights[2], d_i_weights[3]]]
        howeights = [[d_h_weights[0], d_h_weights[1]], [d_h_weights[2], d_h_weights[3]]]
        hidden = Neural_Network.activation_function(input_value, weights)
        output = Neural_Network.activation_function(hidden, howeights)
        actual = list([data['x_axis_output'][i], data['y_axis_output'][i]])
        error = actual[0] - output[0], actual[1] - output[1]
        list_error.append(error)

    test_rmse = RMSE(list_error)

    return test_rmse


def control_the_network(NeuralNetwork_check, data, flag_quarter, flag_half, list_error, j):
    if j / data >= 0.25 and flag_quarter == 0:
        print("here")
        NeuralNetwork_check.decay_lr()
        flag_quarter = 1
        NeuralNetwork_check.regularizer_adjuster(list_error[-2:])
    elif j / data >= 0.5 and flag_half == 0:
        print("Here")
        NeuralNetwork_check.decay_lr()
        flag_half = 1
        NeuralNetwork_check.regularizer_adjuster(list_error[-2:])


def save_best_weights(optimal_weights, error_dict):
    res = {key: optimal_weights[key] for key in sorted(optimal_weights.keys(), key=lambda ele: ele[1] * ele[0])}
    for key, val in res.items():
        # print(key)
        list_storer = val
        # print(list([list_storer[0][0],list_storer[0][1],list_storer[1][0],list_storer[1][1]]))
        break
    with open('best_weights_i_h.pkl', 'wb') as f:
        pickle.dump(list([list_storer[0][0], list_storer[0][1]]), f)
    with open('best_weights_h_o.pkl', 'wb') as f:
        pickle.dump(list([list_storer[1][0], list_storer[1][1]]), f)


def early_stopper(test_error_dict, error_dict, patience=10):
    list_1 = list()
    list_2 = list()
    for x in list(reversed(list(error_dict.values())))[0:patience]:
        list_1.append(x)

    for x in list(reversed(list(test_error_dict.values())))[0:patience]:
        list_2.append(x)
    # print(list_1)
    n = len(list_1) - 2
    count = 0
    for i in range(len(list_1) - 1, 0, -1):
        if i < len(list_1):
            if round(list_1[i][0], 3) <= round(list_1[n][0], 3) or round(list_1[i][1], 3) <= round(list_1[n][1], 3):
                count += 1
        n = n - 1

    n = len(list_2) - 2
    count2 = 0
    for i in range(len(list_2) - 1, 0, -1):
        if i < len(list_2):
            if round(list_2[i][0], 3) <= round(list_1[i][0], 3) * 0.9 or round(list_2[i][1], 3) == round(list_1[i][1],
                                                                                                         3) * 0.9:
                count2 += 1
        n = n - 1

    if count == patience - 1 or count2 == patience - 1:
        return True
    else:
        return False


def main():
    list_error = list()

    data = pd.read_csv("data_ce889.csv")
    data, test = train_test_split(data, test_size=0.3, random_state=42)
    data.drop(['Unnamed: 0'], inplace=True, axis=1)
    # data.drop(['index'], inplace=True, axis=1)
    test.drop(['Unnamed: 0'], inplace=True, axis=1)
    # test.drop(['index'], inplace=True, axis=1)
    data.reset_index(inplace=True)
    test.reset_index(inplace=True)
    print(type(data))
    print(data.head())
    NeuralNetwork_check = NeuralNetwork([2, 2, 2])
    list_lrreg = list()
    # print(normalize([0.8, 0.45]))
    print(NeuralNetwork_check.check())
    input_value = [data['x_axis_input'][0], data['y_axis_input'][0]]
    # weights = [[-0.87, 0.60], [-0.93, -0.086]]
    weights = [[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]]#, [random.uniform(-1, 1), random.uniform(-1, 1)]]
    # howeights = [[-0.81, 0.55], [-0.90, -0.1]]
    howeights = [[random.uniform(-1, 1), random.uniform(-1, 1)],
                 [random.uniform(-1, 1), random.uniform(-1, 1)]]
    # NeuralNetwork_check.weight_multiplier(input_value)
    hidden = NeuralNetwork_check.activation_function(input_value, weights)
    output = NeuralNetwork_check.activation_function(hidden, howeights)
    print(output)
    actual_val = [data['x_axis_output'][0], data['y_axis_output'][0]]
    error = NeuralNetwork_check.error_calc(actual_val, output)
    learning_rate, lambda_reg, error = NeuralNetwork_check.learning_rate, NeuralNetwork_check.lambda_val, error
    list_lrreg.append([learning_rate, lambda_reg, error])
    # list_error.append(error)
    # print(error)
    print("BackProp Begins Here: ")
    o_gradient = NeuralNetwork_check.outgradient(output, error)
    d_h_weights = NeuralNetwork_check.weight_updation(o_gradient, hidden, howeights)
    print(d_h_weights)
    l_gradient = NeuralNetwork_check.local_gradient(hidden, o_gradient, howeights)
    print(l_gradient)
    d_i_weights = NeuralNetwork_check.weight_updation(l_gradient, input_value, weights)
    print(d_i_weights)
    epoch = int(input("How many epochs: "))
    patience = int(input("Patience number for early stopping: "))

    if epoch <= 0:
        epoch = 10

    dict_1 = dict()
    dict_2 = dict()

    optimal_weights = dict()
    # count_quarter = 0
    # count_half = 0
    for i in range(epoch):
        print("----Epoch ", i + 1)
        flag_quarter = 0
        flag_half = 0
        if i >= 1:
            rmse_val = list(dict_1.values())[-1]
            optimal_weights.update({(rmse_val[0], rmse_val[1]): (weights, howeights)})
            if early_stopper(dict_1, dict_2, patience):
                break
            # print(rmse_val)

        for j in range(len(data)):
            if 0.25 <= j / len(data) <= 0.5 and flag_quarter == 0:
                print("here")
                NeuralNetwork_check.decay_lr()
                flag_quarter = 1
                NeuralNetwork_check.regularizer_adjuster(list_error[-2:])
            elif j / len(data) >= 0.5 and flag_half == 0:
                print("Here")
                NeuralNetwork_check.decay_lr()
                flag_half = 1
                NeuralNetwork_check.regularizer_adjuster(list_error[-2:])
            input_value = [data['x_axis_input'][j], data['y_axis_input'][j]]
            weights = [[d_i_weights[0], d_i_weights[1]], [d_i_weights[2], d_i_weights[3]]] # [[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]]
            howeights = [[d_h_weights[0], d_h_weights[1]],
                         [d_h_weights[2], d_h_weights[3]]]  # ,[d_h_weights[4], d_h_weights[5]]]  # only this will increase to accomodate more neurons
            hidden = NeuralNetwork_check.activation_function(input_value, weights)
            output = NeuralNetwork_check.activation_function(hidden, howeights)
            actual_val = [data['x_axis_output'][j], data['y_axis_output'][j]]
            error = NeuralNetwork_check.error_calc(actual_val, output)
            # print("error", error)
            list_error.append(error)
            o_gradient = NeuralNetwork_check.outgradient(output, error)
            d_h_weights = NeuralNetwork_check.weight_updation(o_gradient, hidden, howeights)
            l_gradient = NeuralNetwork_check.local_gradient(hidden, o_gradient, howeights)
            d_i_weights = NeuralNetwork_check.weight_updation(l_gradient, input_value, weights)
        # print(RMSE(list_error))
        dict_1.update({i: RMSE(list_error)})
        test_rmse = test_network(test, d_i_weights, d_h_weights)
        dict_2.update({i: test_rmse})

    learning_rate, lambda_reg, error = NeuralNetwork_check.learning_rate, NeuralNetwork_check.lambda_val, error
    list_lrreg.append([learning_rate, lambda_reg, error])

    with open("lr_lambda.csv", 'a+') as file_handler:
        for item in list_lrreg:
            file_handler.write("{}\n".format(item))
    save_weights(weights, howeights)  # used for saving weights on disk
    save_best_weights(optimal_weights, dict_1)
    # print(weights, howeights)
    in_hid_weights, hid_out_weights = load_weights("best")
    # print(in_hid_weights, hid_out_weights)
    # print(count_quarter, count_half)
    mlt.plot(dict_1.keys(), dict_1.values(), label="train_rmse")
    mlt.plot(dict_1.keys(), dict_2.values(), label="test_rmse")
    mlt.legend()
    # mlt.yscale('log')
    mlt.show()
    # print(inference(data, in_hid_weights, hid_out_weights))

    # print(list_error)


if __name__ == '__main__':
    main()
