import numpy as np
import matplotlib.pyplot as plt
from python_anesthesia_simulator import patient, metrics, Simulator

ts = 60
age = 35
weight = 70
height = 170
sex = 0

# %%

George_1 = patient.Patient([age, height, weight, sex], ts=ts)
George_2 = patient.Patient([age, height, weight, sex], ts=ts)
George_3 = patient.Patient([age, height, weight, sex], ts=ts)
George_4 = patient.Patient([age, height, weight, sex], ts=ts, model_hemo='VitalDB')

simu_1 = Simulator(George_1, disturbance_profil='realistic')
simu_2 = Simulator(George_2, disturbance_profil='simple')
start_step = 20 * 60
end_step = 30 * 60
simu_3 = Simulator(
    George_3,
    disturbance_profil='step',
    arg_disturbance={'start_step': start_step, 'end_step': end_step},
)
simu_4 = Simulator(George_4, disturbance_profil='VitalDB')

# %% Simulation

N_simu = int(60 * 60 / ts)


uP, uR = 0.13, 0.5


for index in range(N_simu):
    simu_1.one_step(uP, uR)
    simu_2.one_step(uP, uR)
    simu_3.one_step(uP, uR)
    simu_4.one_step(uP, uR)


# %% metrics

metric_1 = metrics.compute_control_metrics(
    simu_1.dataframe.loc[:10 * 60 / ts, 'Time'],
    simu_1.dataframe.loc[:10 * 60 / ts, 'BIS'],
    phase='induction'
)
metric_2 = metrics.compute_control_metrics(
    simu_2.dataframe.loc[:10 * 60 / ts, 'Time'],
    simu_2.dataframe.loc[:10 * 60 / ts, 'BIS'],
    phase='induction'
)
metric_3 = metrics.compute_control_metrics(
    simu_3.dataframe['Time'],
    simu_3.dataframe['BIS'],
    phase='total',
    start_step=start_step,
    end_step=end_step
)

metric_1_new = metrics.new_metrics_induction(
    simu_1.dataframe.loc[:10 * 60 / ts, 'Time'].values,
    simu_1.dataframe.loc[:10 * 60 / ts, 'BIS'].values,
)

metric_3_new = metrics.new_metrics_maintenance(
    simu_3.dataframe.loc[10 * 60 / ts:, 'Time'].values,
    simu_3.dataframe.loc[10 * 60 / ts:, 'BIS'].values,
)


# %% test
def test_intubation_effect():
    """Test that the intubation effect is visible on the signals."""
    assert simu_4.dataframe['HR'].iloc[4] < simu_4.dataframe['HR'].iloc[5]
    assert simu_4.dataframe['SAP'].iloc[4] < simu_4.dataframe['SAP'].iloc[6]
    assert simu_4.dataframe['DAP'].iloc[4] < simu_4.dataframe['DAP'].iloc[6]


def test_surgery_effect():
    """Test that the surgery effect is visible on the signals."""
    assert simu_4.dataframe['HR'].iloc[40] < simu_4.dataframe['HR'].iloc[50]
    assert simu_4.dataframe['SAP'].iloc[40] < simu_4.dataframe['SAP'].iloc[46]
    assert simu_4.dataframe['DAP'].iloc[40] < simu_4.dataframe['DAP'].iloc[46]


def test_metrics():
    # No undershoot during induction
    assert metric_1['BIS_NADIR'].iloc[0] > 50
    assert metric_2['BIS_NADIR'].iloc[0] > 50

    assert np.allclose(metric_1['US'].iloc[0], 0.0)
    assert np.allclose(metric_2['US'].iloc[0], 0.0)

    # Time to target equal to 9 minutes
    assert np.allclose(metric_1['TT'].iloc[0], 9)
    assert np.allclose(metric_2['TT'].iloc[0], 9)
    assert np.allclose(metric_3['TT'].iloc[0], 9)

    # Settling time at 10% equal to 9 minutes
    assert np.allclose(metric_1['ST10'].iloc[0], 9)
    assert np.allclose(metric_2['ST10'].iloc[0], 9)
    assert np.allclose(metric_3['ST10'].iloc[0], 9)

    # Settling time at 20% equal to 6 minutes
    assert np.allclose(metric_1['ST20'].iloc[0], 6)
    assert np.allclose(metric_2['ST20'].iloc[0], 6)
    assert np.allclose(metric_3['ST20'].iloc[0], 6)

    # test maintenance phase
    assert np.allclose(metric_3['TTp'].iloc[0], 10)
    assert metric_3['BIS_NADIRp'].iloc[0] > 50
    assert np.isnan(metric_3['TTn'].iloc[0])
    assert metric_3['BIS_NADIRn'].iloc[0] < 50


def test_new_metrics():
    """test the new metrics."""

    assert (metric_1_new['Sleep_Time'] == 6).all()
    assert (metric_1_new["Low BIS time"] == 0).all()
    assert abs(metric_1_new['Lowest BIS'].iloc[0] - 53.2) <= 1e-1
    assert (metric_1_new['Settling time'] == 6).all()

    assert (metric_3_new['Time out of range'] == 0).all()
    assert abs(metric_3_new['Lowest BIS'].iloc[0] - 42.2) <= 1e-1
    assert abs(metric_3_new['Highest BIS'].iloc[0] - 57.0) <= 1e-1

# %% plots


if __name__ == '__main__':
    Time = simu_1.dataframe['Time'] / 60

    fig, ax = plt.subplots(3)
    ax[0].plot(Time, simu_1.dataframe['u_propo'])
    ax[1].plot(Time, simu_1.dataframe['u_remi'])
    ax[2].plot(Time, simu_1.dataframe['u_nore'])

    ax[0].set_ylabel("Propo")
    ax[1].set_ylabel("Remi")
    ax[2].set_ylabel("Nore")

    plt.show()

    simu_4.disturbances.plot_dist()

    fig, ax = plt.subplots(4)

    ax[0].plot(Time, simu_1.dataframe['BIS'])
    ax[1].plot(Time, simu_1.dataframe['MAP'])
    ax[2].plot(Time, simu_1.dataframe['CO'])
    ax[3].plot(Time, simu_1.dataframe['HR'])
    ax[0].plot(Time, simu_2.dataframe['BIS'])
    ax[1].plot(Time, simu_2.dataframe['MAP'])
    ax[2].plot(Time, simu_2.dataframe['CO'])
    ax[3].plot(Time, simu_2.dataframe['HR'])
    ax[0].plot(Time, simu_3.dataframe['BIS'])
    ax[1].plot(Time, simu_3.dataframe['MAP'])
    ax[2].plot(Time, simu_3.dataframe['CO'])
    ax[3].plot(Time, simu_3.dataframe['HR'])
    ax[0].plot(Time, simu_4.dataframe['BIS'])
    ax[1].plot(Time, simu_4.dataframe['SAP'])
    ax[2].plot(Time, simu_4.dataframe['CO'])
    ax[3].plot(Time, simu_4.dataframe['HR'])

    ax[0].set_ylabel("BIS")
    ax[1].set_ylabel("MAP")
    ax[2].set_ylabel("CO")
    ax[3].set_ylabel("HR")
    ax[3].set_xlabel("Time (min)")
    for i in range(4):
        ax[i].grid()
    plt.show()

    test_metrics()
    test_new_metrics()
    test_intubation_effect()
    test_surgery_effect()
    print('All tests passed successfully!')
