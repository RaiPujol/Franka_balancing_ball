"""Render selected estimated-feedback demonstrations on the STL scene."""
from stl_bridge import use_actual_stl_scene
from stl_scene import MEASURED_PRINT_MASS_KG, PLATE_SIDE_MM
import render_methods
import render_selected


if __name__=='__main__':
    # Entry points share this label dictionary. The true CAD is visible, while
    # the support contact is an explicit flat approximation; see README.
    render_methods.LABELS['lqr']=f'{PLATE_SIDE_MM:g} mm / {1000 * MEASURED_PRINT_MASS_KG:g} g | Kalman + LQR'
    with use_actual_stl_scene():
        render_selected.main()
