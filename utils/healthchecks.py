from utils.arrays import get_model_shape


def check_models_consistency(model_repr_dict):
    try:
        for (_, model_repr_list) in model_repr_dict.items():
            # Create reference model shape from the first model
            reference = get_model_shape(model_repr_list[0])

            # Ensure every model has the same shape as the reference model
            for model_repr in model_repr_list:
                assert len(get_model_shape(model_repr) ^ reference) == 0
        print("Models have consistent shapes!")
    except AssertionError:
        print("Shape inconsistency detected!")
