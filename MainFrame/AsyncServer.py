import random
import copy


class AsyncServer:
    def __init__(self, global_model, c_ratio, client_list, init_weight, valid_data, test_data, batch_size,
                 select_list):
        self.global_model = global_model
        self.client_ratio = c_ratio
        self.client_list = client_list
        self.clients_num = len(client_list)
        self.init_server_weights = init_weight
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.select_list = select_list

        self.accuracy_list = []
        self.loss_list = []

    def asynchronous_train(self, epoch):

        # 初始化全局模型
        self.global_model.set_weights(self.init_server_weights)

        # 开始 epoch 轮的异步联邦学习训练
        for e in range(epoch):
            # 选择一定比率的客户端
            selected_clients = random.sample(self.client_list, int(self.clients_num * self.client_ratio))

            # 将全局模型发送给选中的客户端
            for s_client in selected_clients:
                s_client.receive_global_model(copy.deepcopy(self.global_model.get_weights()))

            # 将客户端返回的更新与全局模型进行聚合，从而更新全局模型
            client_count = 0
            for client in self.client_list:
                if client.is_ready_to_return():
                    part_layer_client_weights, select_list = client.return_weights()
                    self.update_global_model(part_layer_client_weights, select_list, 0.2)

                    # 测试并记录全局模型的性能
                    accuracy, loss = self.global_model.evaluate_network(self.test_data[0], self.test_data[1])
                    self.accuracy_list.append(accuracy)
                    self.loss_list.append(loss)
                    client_count += 1
                    print("Train Epoch", e, " client ", client_count, ":  accuracy =", accuracy, ", loss =", loss)

    def update_global_model(self, client_weights, select_list, alpha):
        updated_weights = []
        server_weights = copy.deepcopy(self.global_model.get_weights())
        for layer in range(len(server_weights)):
            if select_list[layer] is True:  # 本层传输了，就进行聚合
                updated_weights.append(((1 - alpha) * server_weights[layer]) + (alpha * client_weights[layer]))
            else:  # 本层没传输，就进行保持全局模型本次参数不变
                updated_weights.append(server_weights[layer])

        self.global_model.set_weights(updated_weights)

    def get_accuracy_and_loss_result(self):
        return self.accuracy_list, self.loss_list
