import copy
from models.CSFINet import CSFINet


def get_model(model_dict, n_classes, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name == "CSFINet":
        model = model(input_channel=3, n_classes=n_classes, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True, is_deconv=True, is_batchnorm=True, selfeat=True, shift_n=5, auxseg=True)


    return model


def _get_model_instance(name):
    try:
        return {
            "CSFINet": CSFINet
        }[name]
    except:
        raise ("Model {} not available".format(name))

