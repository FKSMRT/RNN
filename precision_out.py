import os
import numpy as np
import pandas as pd

def p_out(subject):
    test_subject = subject

    dir = os.path.expanduser("~")
    path = "Python\B4_1\RNN\LF_result\Models precision"

    maepath = "MAE\MAE_['" + test_subject + "'].npy"
    rmsepath = "RMSE\RMSE_['" + test_subject + "'].npy"
    scorepath = "score\score_['" + test_subject + "'].npy"

    dirpath_mae = os.path.join(dir, path, maepath)
    dirpath_rmse = os.path.join(dir, path, rmsepath)
    dirpath_score = os.path.join(dir, path, scorepath)

    mae = np.load(dirpath_mae)
    rmse = np.load(dirpath_rmse)
    score = np.load(dirpath_score)


    ############################## 以下 スコア出力 ###############################
    test_label = ["Only Solvent", "Low Concentration", "High Concentration"]
    mscore = np.mean(score)

    print("\n")
    print("RMSE \n " + str(rmse))
    print("MAE \n" + str(mae))
    print("Score (%) \n" + str(test_label[:]) + " = " + str(score))
    print("Score (%) \n" + str(mscore) )
    print("\n")

    return rmse , mscore


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

subjects = ["A", "B", "C", "D", "E", "F", "G", "H"]
dir = os.path.expanduser("~")
savepath = os.path.join(dir, "Python\B4_1\RNN\\to_csv")

rmse = []
mscore = []

for i in subjects:
    print("subject = ", i)
    _rmse, _mscore = p_out(i)
    rmse += [_rmse]
    mscore += [_mscore]



print(rmse)
print(mscore)

if not os.path.exists(savepath):
    os.mkdir(savepath)

np.savetxt(savepath + "\\RMSE_LF.csv", rmse, delimiter=",")
np.savetxt(savepath + "\\Score_LF.csv", mscore, delimiter=",")
