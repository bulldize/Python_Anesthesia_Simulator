clear all
close all
clc

env = pyenv('Version', ...
    'C:\Users\michele.schiavo\python_anesthesia_simulator_env\Scripts\python.exe');

simulation_time = 3600;

age = 18;
height = 170;
weight = 60;
gender = 0;

sampling_time = 1;

y = sim('test_pythonPatient.slx');

subplot(3,1,1)
title('BIS')
plot(y.bis, 'k', 'LineWidth', 1.2)
hold on
grid on
ylabel('BIS')
xlabel('Time [s]')


% Plot Propofol infusion rate
subplot(3,1,2)
title('Propofol Infusion Rate')
plot(y.u_prop, 'k', 'LineWidth', 1.2)
hold on
grid on
xlabel('Time [s]')
ylabel('Infusion Rate [mg/s]')

% Plot Remifentanil infusion rate
subplot(3,1,3)
title('Remifentanil Infusion Rate')
plot(y.u_remi, 'k', 'LineWidth', 1.2)
hold on
grid on
xlabel('Time [s]')
ylabel('Infusion Rate [ug/s]')


