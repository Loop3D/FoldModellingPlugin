import ipywidgets as widgets
from IPython.display import display, clear_output

# Initial dictionary structure
dict_structure = {
    'fold_limb_rotation_angle': {},
    'fold_axis_rotation_angle': {},
    'fold_axial_surface': {}
}


def create_mu_widgets():
    mu_widgets = [
        widgets.FloatText(value=0, description=f'mu[{i}]') for i in range(3)
    ]
    return widgets.HBox(mu_widgets)


def create_value_widgets():
    selected_constraint = constraint_dropdown.value
    selected_sub_constraint = sub_constraint_dropdown.value
    if selected_constraint == 'fold_axial_surface' and selected_sub_constraint == 'axial_surface':
        mu_widget = create_mu_widgets()
        return {
            'mu': mu_widget,
            'kappa': widgets.FloatText(value=5, description='kappa'),
            'w': widgets.FloatText(value=1, description='w')
        }
    else:
        mu_widget = widgets.FloatText(value=0, description='mu')
    return {
        'mu': mu_widget,
        'sigma': widgets.FloatText(value=0, description='sigma'),
        'w': widgets.FloatText(value=1, description='w')
    }


def on_add_button_click(button):
    selected_constraint = constraint_dropdown.value
    selected_sub_constraint = sub_constraint_dropdown.value
    values = {}
    for k, v in value_widgets.items():
        if isinstance(v, widgets.HBox):
            # Assuming mu is the only HBox and is always composed of three FloatText widgets
            values[k] = [w.value for w in v.children]
        else:
            values[k] = v.value
    dict_structure[selected_constraint][selected_sub_constraint] = values
    with output:
        clear_output()
        print(dict_structure)


def on_constraint_change(change):
    new_value = change.get('new', None)
    sub_constraint_dropdown.options = sub_constraints.get(new_value, [])
    # Only call on_sub_constraint_change if form is defined
    if 'form' in globals():
        on_sub_constraint_change({'new': sub_constraint_dropdown.value})


def on_sub_constraint_change(change):
    global value_widgets
    value_widgets = create_value_widgets()
    form.children = [constraint_dropdown, sub_constraint_dropdown] + list(value_widgets.values()) + [add_button, output]


# Mapping of Major Constraints to their sub-constraints
sub_constraints = {
    'fold_limb_rotation_angle': ['tightness', 'asymmetry', 'fold_wavelength', 'axial_trace_1', 'axial_traces_2',
                                 'axial_traces_3', 'axial_traces_4'],
    'fold_axis_rotation_angle': ['hinge_angle', 'fold_axis_wavelength'],
    'fold_axial_surface': ['axial_surface'],
}


def display_dict_selection():
    # Dropdown for constraints
    constraint_dropdown = widgets.Dropdown(options=list(sub_constraints.keys()), description='Major Constraint:')
    constraint_dropdown.observe(on_constraint_change, names='value')

    # Dropdown for sub-constraints
    sub_constraint_dropdown = widgets.Dropdown(description='Sub-Constraint:')
    sub_constraint_dropdown.observe(on_sub_constraint_change, names='value')

    # Button to add the details
    add_button = widgets.Button(description="Add Details")
    add_button.on_click(on_add_button_click)

    # Output widget to display the generated dictionary
    output = widgets.Output()

    # Initial value widgets
    value_widgets = create_value_widgets()

    # Form to hold all the widgets
    form = widgets.VBox(
        [constraint_dropdown, sub_constraint_dropdown] + list(value_widgets.values()) + [add_button, output])

    # Initial setup
    on_constraint_change({'new': constraint_dropdown.value})

    display(form)
