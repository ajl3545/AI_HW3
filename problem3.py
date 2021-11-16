# Alex Lamarche
# AI EXP HW3 : Problem 3 : problem3.py

from numpy.core.fromnumeric import transpose
import pandas as pd
import numpy as np
import problem1 as p1

import matplotlib.pyplot as plt


def load_table(table, sheet):
    data = pd.read_excel(table, sheet_name=sheet)
    return data


def execute(data, c):

    # Get the X_tr matrix
    X_tr = data.iloc[:, 1:].values.tolist()
    # Get the y_tr vector
    y_tr = np.array(data.iloc[:, 0].values.tolist())

    l = [0, 0.1, 1]

    color_count = 0
    for l_i in l:

        (Wopt, mtr) = p1.CF_SOLVER(X_tr, y_tr, l_i)
        print("Wopt = \n" + str(Wopt.tolist()))
        print("Opt Cost (mtr) = " + str(mtr))

        mse = p1.MSE(X_tr, y_tr, Wopt)
        print("Opt MSE = " + str(mse))

        # 3.2 Intermediate MSE's during descent
        (W, costs, norms) = p1.GD_SOLVER(
            X_tr, y_tr, np.ones(len(X_tr[0])+1), l_i, 0.001)
        mse_count = 0

        ms = []

        for w_params in W:
            m = p1.MSE(X_tr, y_tr, w_params)
            ms.append(m)
            m_s = "{:.10f}".format(m)
            print("MSE " + str(mse_count) + " = " + m_s)
            mse_count += 1

        # plot the points

        plt.title(str("l=[0,0.1,1]"))
        plt.plot(np.transpose([range(len(W))]), ms, color=c[color_count])
        plt.plot(np.transpose([range(len(W))]), [
                 mtr]*len(W), color=c[color_count])
        color_count += 1


def execute_first_10(data, c):

    # Get the X_tr matrix
    X_tr = data.iloc[:10, 1:].values.tolist()
    # Get the y_tr vector
    y_tr = data.iloc[:10, 0].values.tolist()
    # lambda = 0 for problem 2.2
    l = [0, 0.1, 1]

    color_count = 0
    for l_i in l:

        (Wopt, mtr) = p1.CF_SOLVER(X_tr, y_tr, l_i)
        print("Wopt = \n" + str(Wopt.tolist()))
        print("Opt Cost (mtr) = " + str(mtr))

        mse = p1.MSE(X_tr, y_tr, Wopt)
        print("Opt MSE = " + str(mse))

        # 3.2 Intermediate MSE's during descent
        (W, costs, norms) = p1.GD_SOLVER(
            X_tr, y_tr, np.ones(len(X_tr[0])+1), l_i, 0.001)
        mse_count = 0

        ms = []

        for w_params in W:
            m = p1.MSE(X_tr, y_tr, w_params)
            ms.append(m)
            m_s = "{:.10f}".format(m)
            print("MSE " + str(mse_count) + " = " + m_s)
            mse_count += 1

        # plot the points

        plt.title(str("l=[0,0.1,1]"))
        plt.plot(np.transpose([range(len(W))]), ms, color=c[color_count])
        plt.plot(np.transpose([range(len(W))]), [
                 mtr]*len(W), color=c[color_count])
        color_count += 1


# 2.1 loading data from table
noisy_tr_data = load_table("data/noisy_data.xlsx", "Sheet1")
noisy_v_data = load_table("data/noisy_data.xlsx", "Sheet2")
noisy_te_data = load_table("data/noisy_data.xlsx", "Sheet3")

# Training executions
# execute(noisy_tr_data, ["red", "green", "blue"])           # all data
execute_first_10(noisy_tr_data, ["red", "green", "blue"])   # first 10 rows

# Testing executions
# execute(noisy_te_data, ["red", "green", "blue"])            # all data
#execute_first_10(noisy_te_data,["red", "green", "blue"])

plt.show()
