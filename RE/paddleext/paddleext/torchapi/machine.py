"""
machine for paddle
"""

import paddle


class PaddleTrainer(object):
    """
    PaddleTrainer
    """

    def __init__(self, machine, loss, optimizer,
                 evaluator, *args, **kwargs):

        self.model = paddle.Model(machine)

        self.model.prepare(optimizer=optimizer, loss=loss,
                           metrics=evaluator)

    def fit(self, train_data_streams):
        """

        Args:
            train_dataloader ():
            val_dataloaders ():
            test_dataloaders ():

        Returns:

        """

        self.model.fit(train_data_streams.train, eval_data=train_data_streams.dev)

Trainer = PaddleTrainer