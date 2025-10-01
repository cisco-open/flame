import logging

from examples.fwdllm.trainer.forward_training.model_trainer import ModelTrainer


class FedTransformerTrainer(ModelTrainer):

    def __init__(self, trainer, model):
        super().__init__(model)
        self.model_trainer = trainer
        self.model = model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def get_model(self):
        return self.model

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info(
            "Client(%d)" % self.id + ":| Local Train Data Size = %d" % (len(train_data))
        )
        logging_state = {"round_id": None, "data_id": None}
        self.model_trainer.train_dl = train_data
        self.model_trainer.train_model(device=device, logging_state=logging_state)

    def test(self, test_data, device, args=None):
        pass

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ):
        self.model_trainer.eval_model(device=device)
        return True

    # 在前test_num个batch上进行快速的test
    def quick_test(self, device, test_num=1000):
        acc = self.model_trainer.quick_test(device=device, test_num=test_num)
        return acc

    def quick_test_loss(self, device, test_num=1000):
        loss = self.model_trainer.quick_test_loss(device=device, test_num=test_num)
        return loss
