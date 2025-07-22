import matplotlib.pyplot as plt
import numpy as np

from python_anesthesia_simulator import simulator


# Simulation duration in seconds
Tsim = 3600
# Patient physical data
age = 20
weight = 70
height = 170
gender = 0

ts = 1

# %% Baseline simulation

Nsim = int(Tsim/ts)

# Initialize infusion profiles for propofol, remifentanil and norepinephrine
propofol_infusion_profile = np.zeros((Nsim,))        # mg/s
remifentanil_infusion_profile = np.zeros((Nsim,))    # ug/s
norepinephrine_infusion_profile = np.zeros((Nsim,))    # ug/s

# Initialize sqi profile
sqi = 50

# Propofol profile
propofol_infusion_profile[0:int(50/ts)] = 2          # 2 mg/s for 50 seconds
propofol_infusion_profile[int(150/ts):] = 0.2        # 0.2 mg/s from 150s onward

# Young patient

bis_delay_1 = 120 * (1 - sqi/100)
George_1 = simulator.Patient([age, height, weight, gender],
                             ts=ts,
                             random_PD=False)

# One step simulation
for k in range(Nsim-1):
    uProp_k = propofol_infusion_profile[k]    
    George_1.one_step(u_propo=uProp_k, sqi=sqi, noise=False)



# Compute the delays obtained in simulation


change = George_1.dataframe['BIS'] != George_1.dataframe['BIS'].shift()
first_change_index = George_1.dataframe['BIS'][change].index[1]
simulated_delay_1 = George_1.dataframe['Time'][first_change_index]




# %% plot
if __name__ == '__main__':
    fig, ax = plt.subplots(2)

    ax[0].plot(George_1.dataframe['Time'], George_1.dataframe['BIS'])

    ax[1].plot(George_1.dataframe['Time'], George_1.dataframe['u_propo'])


    ax[0].set_ylabel("BIS")
    ax[1].set_ylabel("Propofol infusion")
    ax[1].set_xlabel("Time (min)")
    for i in range(2):
        ax[i].grid()
    plt.ticklabel_format(style='plain')
    plt.show()

# %%
def test_delay():
    """Check that the BIS delays obtained in the simulations match those expected"""

    assert simulated_delay_1 >= np.floor(bis_delay_1)
