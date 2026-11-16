"""Render selected estimated-feedback demonstrations on the STL scene."""
from stl_bridge import use_actual_stl_scene
import render_methods
import render_selected


if __name__=='__main__':
    # Entry points share this label dictionary. The true CAD is visible, while
    # the support contact is an explicit flat approximation; see README.
    render_methods.LABELS['lqr']='STL visual / flat contact | Kalman + LQR'
    with use_actual_stl_scene():
        render_selected.main()
