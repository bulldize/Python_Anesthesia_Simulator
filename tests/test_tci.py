import matplotlib.pyplot as plt
import numpy as np
from python_anesthesia_simulator import Patient, TCIController, Simulator


age = 28
height = 165
weight = 65
gender = 0
patient_info = [age, height, weight, gender]

sampling_time = 2
propofol_target = 4
remifentanil_target = 3


# init the patient simulation
patient = Patient(
    patient_info,
    ts=sampling_time,
    model_propo='Schnider',
    model_remi='Minto'
)
simulator_wo_tci = Simulator(patient)
patient_2 = Patient(
    patient_info,
    ts=sampling_time,
    model_propo='Schnider',
    model_remi='Minto'
)

simulator_w_tci = Simulator(patient_2,
                            tci_propo='Effect_site',
                            tci_remi='Effect_site')

# initialize TCI
tci_propo = TCIController(
    patient_info,
    sampling_time=sampling_time,
    drug_name='Propofol',
    model_used="Schnider",
)
tci_remi = TCIController(
    patient_info,
    sampling_time=sampling_time,
    drug_name='Remifentanil',
    model_used="Minto",
)

N_simu = 5 * 60 // sampling_time  # 10 minutes


for time_step in range(N_simu):
    u_propo = tci_propo.one_step(propofol_target)
    u_remi = tci_remi.one_step(remifentanil_target)

    simulator_wo_tci.one_step(u_propo, u_remi)
    simulator_w_tci.one_step(
        input_propo=propofol_target,
        input_remi=remifentanil_target
    )
# test


def test_tci_ouput_range():
    """ensure that the command belong in the acceptable range."""
    assert (simulator_wo_tci.dataframe['u_propo'] >= 0).all()
    assert (simulator_wo_tci.dataframe['u_propo'] <= tci_propo.infusion_max).all()
    assert (simulator_wo_tci.dataframe['u_remi'] >= 0).all()
    assert (simulator_wo_tci.dataframe['u_remi'] <= tci_remi.infusion_max).all()


def test_tci_behavior():
    # ensure that the concentration reach the target (maximum of 1%)
    assert simulator_wo_tci.dataframe['x_propo_4'].iloc[-1] <= propofol_target * 1.01
    assert simulator_wo_tci.dataframe['x_propo_4'].iloc[-1] >= propofol_target * 0.99
    assert simulator_wo_tci.dataframe['x_remi_4'].iloc[-1] <= remifentanil_target * 1.01
    assert simulator_wo_tci.dataframe['x_remi_4'].iloc[-1] >= remifentanil_target * 0.99

    # ensure that there is not too much overshoot (maximum 5%)
    assert (simulator_wo_tci.dataframe['x_propo_4'].iloc[-1] <= propofol_target * 1.05).all()
    assert (simulator_wo_tci.dataframe['x_remi_4'].iloc[-1] <= remifentanil_target * 1.05).all()


def test_simulator_tci():
    # ensure that both simulation are the same
    for signal in ['u_propo', 'u_remi', 'x_propo_1', 'x_remi_1']:
        assert np.allclose(simulator_wo_tci.dataframe[signal], simulator_w_tci.dataframe[signal])


if __name__ == "__main__":
    plt.subplot(2, 1, 1)
    plt.plot(simulator_wo_tci.dataframe['Time'] / 60, simulator_wo_tci.dataframe['u_propo'], label='Propofol (mg/s)')
    plt.plot(simulator_wo_tci.dataframe['Time'] / 60, simulator_wo_tci.dataframe['u_remi'], label='Remifentanil (ug/s)')
    plt.ylabel('Drug rate')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(simulator_wo_tci.dataframe['Time'] / 60, simulator_wo_tci.dataframe['x_propo_4'], label='Propofol (ug/ml)')
    plt.plot(simulator_wo_tci.dataframe['Time'] / 60,
             simulator_wo_tci.dataframe['x_remi_4'], label='Remifentanil (ng/ml)')
    plt.plot(simulator_w_tci.dataframe['Time'] / 60,
             simulator_w_tci.dataframe['x_propo_4'], '--', label='Propofol in simulator (ug/ml)')
    plt.plot(simulator_w_tci.dataframe['Time'] / 60,
             simulator_w_tci.dataframe['x_remi_4'], '--', label='Remifentanil in simulator (ng/ml)')
    # plot target
    plt.plot(simulator_wo_tci.dataframe['Time'] / 60, [propofol_target] *
             len(simulator_wo_tci.dataframe['Time']), '--', label='Propofol target')
    plt.plot(simulator_wo_tci.dataframe['Time'] / 60, [remifentanil_target] *
             len(simulator_wo_tci.dataframe['Time']), '--', label='Remifentanil target')
    plt.ylabel('Effect site concentration ')
    plt.xlabel('Time (min)')
    plt.legend()
    plt.grid()

    plt.show()

    test_tci_ouput_range()
    test_tci_behavior()
    test_simulator_tci()
    print("All tests passed successfully.")
