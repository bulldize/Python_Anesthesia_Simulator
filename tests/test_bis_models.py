from python_anesthesia_simulator.pd_models import BIS_model

# BIS_model object to test the Bouillon propofol-remifentanil interaction model
bouillon_bis_model = BIS_model('Bouillon')
#bouillon_bis_model.plot_surface()

# BIS_model object to test the Vanluchene propofol model
vanluchene_bis_model = BIS_model('Vanluchene')
#vanluchene_bis_model.plot_surface()

# BIS_model object to test the Eleveld propofol model
eleveld_bis_model = BIS_model('Eleveld',age = 70)
#eleveld_bis_model.plot_surface()

# BIS_model object to test the Fuentes propofol-remifentanil interaction model
fuentes_bis_model = BIS_model('Fuentes')
#fuentes_bis_model.plot_surface()

# BIS_model object to test the Kern propofol-remifentanil model
kern_bis_model = BIS_model('Kern')
#kern_bis_model.plot_surface()

# BIS_model object to test the Mertens propofol-remifentanil model
mertens_bis_model = BIS_model('Mertens')
#mertens_bis_model.plot_surface()

# BIS_model object to test the Johnson propofol-remifentanil model
johnson_bis_model = BIS_model('Johnson')
#johnson_bis_model.plot_surface()

# BIS_model object to test the Yumuk propofol-remifentanil model
yumuk_bis_model = BIS_model('Yumuk')
#yumuk_bis_model.plot_surface()

# Create a BIS_model object initialized with custom parameters equal to those of bouillon
custom_bis_model_bouillon = BIS_model(hill_param = [4.47, 19.3, 1.43, 0, 97.4, 97.4])

# Create a BIS_model object initialized with custom parameters equal to those of vanluchene
custom_bis_model_vanluchene = BIS_model(hill_param = [4.92, 0, 2.69, 0, 95.9, 87.5])


# tests
def test_default_initialization():
    """Ensure that the default models give correct results"""
    # Check results at low concentrations
    assert vanluchene_bis_model.compute_bis(0) >= 90
    assert eleveld_bis_model.compute_bis(0) >= 90
    assert bouillon_bis_model.compute_bis(0,0) >= 90
    assert fuentes_bis_model.compute_bis(0,0) >= 90
    assert kern_bis_model.compute_bis(0,0) >= 90
    assert mertens_bis_model.compute_bis(0,0) >= 90
    assert johnson_bis_model.compute_bis(0,0) >= 90
    assert yumuk_bis_model.compute_bis(0,0) >= 90
    # Check results at high concentrations
    assert vanluchene_bis_model.compute_bis(16) <= 20
    assert eleveld_bis_model.compute_bis(16) <= 20
    assert bouillon_bis_model.compute_bis(12,8) <= 20
    assert fuentes_bis_model.compute_bis(12,8) <= 20
    assert kern_bis_model.compute_bis(12,8) <= 20
    assert mertens_bis_model.compute_bis(12,8) <= 20
    assert johnson_bis_model.compute_bis(12,8) <= 20
    assert yumuk_bis_model.compute_bis(12,8) <= 20
    # Check results at clinically recommended concentrations
    assert vanluchene_bis_model.compute_bis(5) <= 60
    assert vanluchene_bis_model.compute_bis(5) >= 40
    assert eleveld_bis_model.compute_bis(2.5) <= 60
    assert eleveld_bis_model.compute_bis(2.5) >= 40
    assert bouillon_bis_model.compute_bis(3,6) <= 60
    assert bouillon_bis_model.compute_bis(3,6) >= 40
    assert fuentes_bis_model.compute_bis(3,6) <= 60
    assert fuentes_bis_model.compute_bis(3,6) >= 40
    assert kern_bis_model.compute_bis(1,1.5) <= 60
    assert kern_bis_model.compute_bis(1,1.5) >= 40
    assert mertens_bis_model.compute_bis(1.5,3) <= 60
    assert mertens_bis_model.compute_bis(1.5,3) >= 40
    assert johnson_bis_model.compute_bis(1.5,3) <= 60
    assert johnson_bis_model.compute_bis(1.5,3) >= 40
    assert yumuk_bis_model.compute_bis(4,8) <= 60
    assert yumuk_bis_model.compute_bis(4,8) >= 40
    
def test_custom_initialization():
    """Ensure that the custom models give correct results by comparing with 
    the results given by the default models"""
    # Check results at low concentrations
    assert abs(vanluchene_bis_model.compute_bis(0) - 
               custom_bis_model_vanluchene.compute_bis(0)) < 1e-3
    assert abs(bouillon_bis_model.compute_bis(0,0) - 
               custom_bis_model_bouillon.compute_bis(0,0)) < 1e-3
    # Check results at high concentrations
    assert abs(vanluchene_bis_model.compute_bis(16) - 
               custom_bis_model_vanluchene.compute_bis(16)) < 1e-3
    assert abs(bouillon_bis_model.compute_bis(12,8) - 
               custom_bis_model_bouillon.compute_bis(12,8)) < 1e-3
    # Check results at clinically recommended concentrations
    assert abs(vanluchene_bis_model.compute_bis(5) - 
               custom_bis_model_vanluchene.compute_bis(5)) < 1e-3
    assert abs(bouillon_bis_model.compute_bis(3,6) - 
               custom_bis_model_bouillon.compute_bis(3,6)) < 1e-3

    
def test_inverse_hill():
    """Check that the inversion of the hill function is giving correct results"""
    # Check results at low concentrations
    assert abs(vanluchene_bis_model.inverse_hill
               (vanluchene_bis_model.compute_bis(0)) - 0) < 1e-3
    assert abs(eleveld_bis_model.inverse_hill
               (eleveld_bis_model.compute_bis(0)) - 0) < 1e-3
    assert abs(bouillon_bis_model.inverse_hill
               (bouillon_bis_model.compute_bis(0,0),0) - 0) < 1e-3
    # Check results at high concentrations
    assert abs(vanluchene_bis_model.inverse_hill
               (vanluchene_bis_model.compute_bis(16)) - 16) < 1e-3
    assert abs(eleveld_bis_model.inverse_hill
               (eleveld_bis_model.compute_bis(16)) - 16) < 1e-3
    assert abs(bouillon_bis_model.inverse_hill
               (bouillon_bis_model.compute_bis(12,8),8) - 12) < 1e-3
    # Check results at clinically recommended concentrations
    assert abs(vanluchene_bis_model.inverse_hill
               (vanluchene_bis_model.compute_bis(5)) - 5) < 1e-3
    assert abs(eleveld_bis_model.inverse_hill
               (eleveld_bis_model.compute_bis(2)) - 2) < 1e-3
    assert abs(eleveld_bis_model.inverse_hill
               (eleveld_bis_model.compute_bis(4)) - 4) < 1e-3
    assert abs(bouillon_bis_model.inverse_hill
               (bouillon_bis_model.compute_bis(3,6),6) - 3) < 1e-3
    assert abs(fuentes_bis_model.inverse_hill
               (fuentes_bis_model.compute_bis(3,6),6) - 3) < 1e-3
    assert abs(kern_bis_model.inverse_hill
               (kern_bis_model.compute_bis(1,1.5),1.5) - 1) < 1e-3
    assert abs(mertens_bis_model.inverse_hill
           (mertens_bis_model.compute_bis(1.5,3),3) - 1.5) < 1e-3
    assert abs(johnson_bis_model.inverse_hill
           (johnson_bis_model.compute_bis(1.5,3),3) - 1.5) < 1e-3
    assert abs(yumuk_bis_model.inverse_hill
           (yumuk_bis_model.compute_bis(4,8),8) - 4) < 1e-3