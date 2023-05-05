import logging
import functools

from loss.loss import (
    cross_entropy2d
)


logger = logging.getLogger("ptsemseg")

key2loss = {
    "cross_entropy": cross_entropy2d
}


def get_loss_function(cfg):
    if cfg["training"]["loss"] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        # {'name': 'cross_entropy'}
        loss_dict = cfg["training"]["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        # 固定loss_name中的参数
        return functools.partial(key2loss[loss_name], **loss_params)