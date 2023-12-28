import pandas
import matplotlib.pyplot as plt
import numpy as np

def PltLosses(dir_model : str,model_name : str):
    losses = pandas.read_csv(str(dir_model)+"/losses.csv")
    min=np.argmin(list(losses.Validation))
    plt.plot(np.arange(len(losses)),losses["Train"])
    plt.plot(np.arange(len(losses)),losses["Validation"])
    plt.legend(["Train Loss","Validation Loss"])
    plt.plot(min,losses["Validation"][min],'r*')
    plt.title("Losses de train y validación para {}".format(model_name))
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.show()
    print("Epoca de mejor loss: {}, Mejor loss: {}".format(str(min),str(losses["Validation"][min])))