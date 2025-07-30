from python_anesthesia_simulator.pk_models import AtracuriumModel
from python_anesthesia_simulator.pd_models import NMB_model
from python_anesthesia_simulator.simulator import Patient
import numpy as np
import matplotlib.pyplot as plt

# % Test Bolus
# Simulate the response to an initial atracurium bolus of 500 ug/ml
# The bolus is administered directly in the central compartment and it is
# simulated by initializing the states.
# Since the states are concentrations defined in ug/ml the initial bolus dose
# is converted accordinly by dividing it to the volume of the central
# compartment.

# These are not affecting the Ward Weatherley Lago model
age = 35
height = 170
gender = 0

# Parameters of the Ward Weatherley Lago test model 1
weight_test_model_default = 75
V1_test_model_default = 49

# State initialization to simulate the atracurium bolus administration
x0 = np.array([500/V1_test_model_default, 0, 0, 0])

# Create dictionaries of models parameters that matches the default to check
# that the two initialization methodologies give the same results
pk_parameters_custom = {
    'V1': 49.0,          # Volume of the central compartment [ml/kg]
    'V2': 157.0,         # Volume of the peripheral compartment [ml/kg] 
    'Cl': 5.5,           # Clearance [ml/min/kg]
    't12_alpha': 2.06,   # First half-life time [min]
    't12_beta': 19.9,    # Second half-life time [min]
    'ke0': 0.1,          # Transfer rate of the first effect-site compartment [1/min]
    'tau': 6.2670        # Time constant of the second effect-site compartment [min]
}
pd_parameters_custom = {
    'C50': 0.625,          # Half effect concentration [ug/ml]
    'gamma': 4.25          # Slope 
}

# Create Atracurium PK model objects
test_model_default = AtracuriumModel([age, height, weight_test_model_default, gender])
test_model_custom = AtracuriumModel([age, height, weight_test_model_default, gender],
                                    model_params= pk_parameters_custom)
# Create Atracurium PD model objects
test_hill_default = NMB_model()
test_hill_custom = NMB_model(hill_param= pd_parameters_custom)
# test_hill_custom.plot_surface()

# Create a Patient object that implements the atracurium models
George_1 = Patient([age, height, weight_test_model_default, gender])


# Simulation parameters
Tsim = 60*100
ts = 1
Nsim = int(Tsim/ts)
atracurium_infusion_profile = np.zeros((Nsim,))        # ug/s
time = np.arange(Tsim)  # Time axis

# Simulate the free response to a bolus administration with default initialization
x_default = test_model_default.full_sim(u= atracurium_infusion_profile, x0= x0)
nmb_default = test_hill_default.compute_nmb(x_default[3])
# Find response peak and its timing
peak_idx_default = np.argmin(nmb_default)
min_nmb_default = nmb_default[peak_idx_default]
peak_time_default = time[peak_idx_default]
# Find  recovery time NMB>= 50
recovery_idx_default = np.where(nmb_default <= 50)[0][-1]
recovery_time_default = time[recovery_idx_default]

# Simulate the free response to a bolus administration with custom initialization
x_custom = test_model_custom.full_sim(u= atracurium_infusion_profile, x0= x0)
nmb_custom = test_hill_custom.compute_nmb(x_custom[3])
# Find response peak and its timing
peak_idx_custom = np.argmin(nmb_custom)
min_nmb_custom = nmb_custom[peak_idx_custom]
peak_time_custom = time[peak_idx_custom]
# Find  recovery time NMB>= 50
recovery_idx_custom = np.where(nmb_custom <= 50)[0][-1]
recovery_time_custom = time[recovery_idx_custom]

# Simulate the free response to a bolus administration with the Patient object
df_george_1 = George_1.full_sim(u_atra=atracurium_infusion_profile,
                  x0_atra=x0)
nmb_george_1 = df_george_1.NMB
# Find response peak and its timing
peak_idx_george_1 = np.argmin(nmb_george_1)
min_nmb_george_1 = nmb_custom[peak_idx_george_1]
peak_time_george_1 = time[peak_idx_george_1]
# Find  recovery time NMB>= 50
recovery_idx_george_1 = np.where(nmb_george_1 <= 50)[0][-1]
recovery_time_george_1 = time[recovery_idx_george_1]

# tests
def test_default_initialization_atracurium():
    """Ensure that the default models give correct results"""
    # Check the Weatherley Hill curve at relevant atracurium concentrations
    assert test_hill_default.compute_nmb(0) == 100
    assert test_hill_default.compute_nmb(0.4) >= 80
    assert test_hill_default.compute_nmb(0.6) >= 50
    assert test_hill_default.compute_nmb(0.8) <= 30
    assert test_hill_default.compute_nmb(1) <= 20
    assert test_hill_default.compute_nmb(2) <= 10
    
    assert test_hill_custom.compute_nmb(0) == 100
    assert test_hill_custom.compute_nmb(0.4) >= 80
    assert test_hill_custom.compute_nmb(0.6) >= 50
    assert test_hill_custom.compute_nmb(0.8) <= 30
    assert test_hill_custom.compute_nmb(1) <= 20
    assert test_hill_custom.compute_nmb(2) <= 10
    
    
    # Check the key characteristic of the average patient response
    assert min_nmb_default < 1
    assert peak_time_default > 500
    assert 3000 <= recovery_time_default <= 4000
    
    assert min_nmb_custom < 1
    assert peak_time_custom > 500
    assert 3000 <= recovery_time_custom <= 4000
    
    assert min_nmb_george_1 < 1
    assert peak_time_george_1 > 500
    assert 3000 <= recovery_time_george_1 <= 4000

# % plot
if __name__ == '__main__':
    
    
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(time, x_default[i], label=f'x{i+1}(t)')
    plt.plot(df_george_1.Time, df_george_1.x_atra_1, '--')
    plt.plot(df_george_1.Time, df_george_1.x_atra_2, '--')
    plt.plot(df_george_1.Time, df_george_1.x_atra_3, '--')
    plt.plot(df_george_1.Time, df_george_1.x_atra_4, '--')
    plt.xlabel('Time (s)')
    plt.ylabel('State Value')
    plt.title('Evolution of States Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    #plt.plot(time, nmb_default, 'k-', linewidth=1, label='NMB')
    #plt.plot(df_george_1.Time, df_george_1.NMB, 'g--', linewidth=1)
    plt.plot(time, nmb_default, 'k-', label='NMB')
    plt.plot(df_george_1.Time, df_george_1.NMB, 'g--')
    # Set axes limits
    plt.xlim(0, 6000)  # X-axis from 0-6000 seconds
    plt.ylim(0, 120)   # Y-axis from 0-120%
    # Style
    plt.xlabel('time (s)', fontsize=12)
    plt.ylabel('NMB (%)', fontsize=12)
    plt.title('Average patient', fontsize=14, pad=20) 
    # Grid and ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, 7000, 1000))  # Major ticks every 1000s
    plt.yticks(np.arange(0, 140, 20))     # Major ticks every 20%
    plt.tight_layout()
    plt.show()
    