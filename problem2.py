# Alex Lamarche
# AI EXP HW3 : Problem 2 : problem2.py

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
    # lambda = 0 for problem 2.2
    l = 0

    # 2.2 Run CF_SOLVER on data
    (Wopt, mtr) = p1.CF_SOLVER(X_tr, y_tr, l)
    print("Wopt = \n" + str(Wopt.tolist()))
    print("Opt Cost (mtr) = " + str(mtr))

    mse = p1.MSE(X_tr, y_tr, Wopt)
    print("Opt MSE = " + str(mse))

    # 2.3 Intermediate MSE's during descent
    (W, costs, norms) = p1.GD_SOLVER(
        X_tr, y_tr, np.ones(len(X_tr[0])+1), l, 0.01)
    mse_count = 0

    ms = []

    for w_params in W:
        m = p1.MSE(X_tr, y_tr, w_params)
        ms.append(m)
        m_s = "{:.10f}".format(m)
        print("MSE " + str(mse_count) + " = " + m_s)
        mse_count += 1

    # plot the points

    plt.plot(np.transpose([range(len(W))]), ms, color=c)
    plt.plot(np.transpose([range(len(W))]), [mtr]*len(W), color="black")

# For the first 10 values


def execute_first_10(data, c):

    # Get the X_tr matrix
    X_tr = data.iloc[:10, 1:].values.tolist()
    # Get the y_tr vector
    y_tr = data.iloc[:10, 0].values.tolist()
    # lambda = 0 for problem 2.2
    l = 0

    # 2.2 Run CF_SOLVER on data
    (Wopt, mtr) = p1.CF_SOLVER(X_tr, y_tr, l)
    print("Wopt = \n" + str(Wopt.tolist()))
    print("Opt Cost (mtr) = " + str(mtr))

    mse = p1.MSE(X_tr, y_tr, Wopt)
    print("Opt MSE = " + str(mse))

    # 2.3 Intermediate MSE's during descent
    (W, costs, norms) = p1.GD_SOLVER(
        X_tr, y_tr, np.ones(len(X_tr[0])+1), l, 0.01)
    mse_count = 0

    ms = []

    for w_params in W:
        m = p1.MSE(X_tr, y_tr, w_params)
        ms.append(m)
        m_s = "{:.10f}".format(m)
        print("MSE " + str(mse_count) + " = " + m_s)
        mse_count += 1

    # plot the points

    plt.plot(np.transpose([range(len(W))]), ms, color=c)
    plt.plot(np.transpose([range(len(W))]), [mtr]*len(W), color="black")

# For the first 10 values


def execute_l_2(data, c):

    # Get the X_tr matrix
    X_tr = data.iloc[:10, 1:].values.tolist()
    # Get the y_tr vector
    y_tr = data.iloc[:10, 0].values.tolist()
    # lambda = 0 for problem 2.2
    l = 2

    # 2.2 Run CF_SOLVER on data
    (Wopt, mtr) = p1.CF_SOLVER(X_tr, y_tr, l)
    print("Wopt = \n" + str(Wopt.tolist()))
    print("Opt Cost (mtr) = " + str(mtr))

    mse = p1.MSE(X_tr, y_tr, Wopt)
    print("Opt MSE = " + str(mse))

    # 2.3 Intermediate MSE's during descent
    (W, costs, norms) = p1.GD_SOLVER(
        X_tr, y_tr, np.ones(len(X_tr[0])+1), l, 0.01)
    mse_count = 0

    ms = []

    for w_params in W:
        m = p1.MSE(X_tr, y_tr, w_params)
        ms.append(m)
        m_s = "{:.10f}".format(m)
        print("MSE " + str(mse_count) + " = " + m_s)
        mse_count += 1

    # plot the points

    plt.plot(np.transpose([range(len(W))]), ms, color=c)
    plt.plot(np.transpose([range(len(W))]), [mtr]*len(W), color="black")


# 2.1 loading data from table
clean_tr_data = load_table("data/clean_data.xlsx", "Sheet1")
clean_v_data = load_table("data/clean_data.xlsx", "Sheet2")
clean_te_data = load_table("data/clean_data.xlsx", "Sheet3")

# Training executions
execute(clean_tr_data, "red")           # all data
# execute_first_10(clean_tr_data,"blue")  # first 10 rows
#execute_l_2(clean_tr_data, "blue")      # lambda = 2

# Testing executions
# execute(clean_te_data, "red")          # all data
# execute_first_10(clean_te_data,"red")
#execute_l_2(clean_te_data, "red")
plt.show()
