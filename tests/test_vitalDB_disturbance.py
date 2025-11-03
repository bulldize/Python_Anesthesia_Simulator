import matplotlib.pyplot as plt

from python_anesthesia_simulator import patient, Simulator

# %% Simulation setup
# Simulation duration in seconds
Tsim = 90 * 60
# Patient physical data
age = 20
weight = 70
height = 170
gender = 0
# Sampling time
ts = 1

Nsim = int(Tsim / ts)

George = patient.Patient(
    [age, height, weight, gender],
    ts=ts,
    model_propo="Eleveld",
    model_remi="Eleveld",
    model_stimuli='VitalDB',
)

simulator = Simulator(
    George,
    tci_propo='Effect_site',
    tci_remi='Effect_site',
)
propo_target = 4
remi_target = 3

for k in range(Nsim - 1):
    simulator.one_step(input_propo=propo_target, input_remi=remi_target)


def test_intubation_effect():
    """Test that the intubation effect is visible on the signals."""
    assert simulator.dataframe['HR'].iloc[3 * 60] < simulator.dataframe['HR'].iloc[5 * 60]
    assert simulator.dataframe['SAP'].iloc[3 * 60] < simulator.dataframe['SAP'].iloc[5 * 60]
    assert simulator.dataframe['DAP'].iloc[3 * 60] < simulator.dataframe['DAP'].iloc[5 * 60]


def test_surgery_effect():
    """Test that the surgery effect is visible on the signals."""
    assert simulator.dataframe['HR'].iloc[40 * 60] < simulator.dataframe['HR'].iloc[42 * 60]
    assert simulator.dataframe['SAP'].iloc[40 * 60] < simulator.dataframe['SAP'].iloc[42 * 60]
    assert simulator.dataframe['DAP'].iloc[40 * 60] < simulator.dataframe['DAP'].iloc[42 * 60]


if __name__ == '__main__':

    fig, ax = plt.subplots(4)
    ax[0].plot(simulator.dataframe['Time'] / 60, simulator.dataframe['BIS'])
    ax[0].set_ylabel('BIS')
    ax[0].grid()
    ax[1].plot(simulator.dataframe['Time'] / 60, simulator.dataframe['HR'])
    ax[1].set_ylabel('HR (bpm)')
    ax[1].grid()
    ax[2].plot(simulator.dataframe['Time'] / 60, simulator.dataframe['SAP'])
    ax[2].set_ylabel('SAP (mmHg)')
    ax[2].grid()
    ax[3].plot(simulator.dataframe['Time'] / 60, simulator.dataframe['DAP'])
    ax[3].set_ylabel('DAP (mmHg)')
    ax[3].set_xlabel('Time (min)')
    ax[3].grid()
    for i in range(4):
        ax[i].axvline(x=3, color='g', linestyle='--')
        ax[i].axvline(x=40, color='r', linestyle='--')
    plt.show()

    test_intubation_effect()
    test_surgery_effect()
    print("All tests passed successfully.")
