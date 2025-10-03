function [bis, co, map, tol, nmb] = callPython(u_p,u_r,u_n,u_a,age,height,weight,gender,sampling_time)

    persistent simulator George

    if isempty(simulator)

        simulator = ...
            py.importlib.import_module...
            ('python_anesthesia_simulator.simulator');

        George = simulator.Patient([age,height,weight,gender],ts = sampling_time);

    end


    simulation_tuple = George.one_step(u_propo=u_p, ...
                                       u_remi=u_r, ...
                                       u_nore=u_n, ...
                                       u_atra=u_a, ...
                                       noise=false);


    simulation_cell = cell(simulation_tuple);


    bis = double(simulation_cell{1});
    co = double(simulation_cell{2});
    map = double(simulation_cell{3});
    tol = double(simulation_cell{4});
    nmb = double(simulation_cell{5});

end