import Models.FC3 as FC3
import Models.CNN as CNN
import Models.VGG13 as VGG13


class ModelGenerator:
    model_name = "FC3"
    data_type = "MNIST"

    def __init__(self, model_name, data_type):
        self.model_name = model_name
        self.data_type = data_type

    def generate_model(self):
        if self.model_name == "FC3":
            global_network = FC3.FC3(self.data_type)
        elif self.model_name == "CNN":
            global_network = CNN.CNN(self.data_type)
        elif self.model_name == "VGG13":
            global_network = VGG13.VGG13(self.data_type)
        else:
            print("Unexpected dataset name!")
            global_network = FC3.FC3(self.data_type)

        return global_network
