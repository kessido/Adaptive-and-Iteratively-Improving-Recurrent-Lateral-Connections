def load_state_dict(model, state_dict):
    from collections import OrderedDict
    try:
        model.load_state_dict(state_dict)
    except:
        try:
            model.load_state_dict(OrderedDict({f'module.{k}':v for k, v in state_dict.items()}))
        except:
            model.load_state_dict(OrderedDict({k[7:]:v for k, v in state_dict.items()}))
