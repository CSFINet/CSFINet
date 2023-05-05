from loader.brainwebLoader import brainwebLoader
from loader.mrbrainsLoader import mrbrainsLoader
from loader.hvsmrLoader import hvsmrLoader
from loader.hyperLoader import hyperLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "BrainWeb": brainwebLoader,
        'Hyper': hyperLoader,
        "MRBrainS": mrbrainsLoader,
        "HVSMR": hvsmrLoader
    }[name]