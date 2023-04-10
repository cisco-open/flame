# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt

# get data
mus = [1.0, 0.1, 0.01, 0.001, 0.0]
acc_filenames = []
loss_filenames = []

for mu in mus:
    acc_filenames.append(f'aggregator/acc_mu{mu}.txt')
    loss_filenames.append(f'aggregator/loss_mu{mu}.txt')

acc_data = []
loss_data = []
for acc_name, loss_name in zip(acc_filenames, loss_filenames):
    acc_file = open(acc_name, 'r')
    acc_list = []
    for line in acc_file:
        acc_list.append(float(line))
    acc_file.close()
    acc_data.append(acc_list)
    
    loss_file = open(loss_name, 'r')
    loss_list = []
    for line in loss_file:
        loss_list.append(float(line))
    loss_data.append(loss_list)
    loss_file.close()

# save images
def save_figure(acc_data, loss_data, mus):
    X = range(1,101)
    
    # plot mu=0 first
    figure = plt.figure()
    plt.title("FedProx Test Accuracy vs. Rounds (E=4)")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.plot(X, acc_data[-1], label=f"mu={mus[-1]}")
    
    for i in range(len(mus)-1):
        plt.plot(X, acc_data[i], label=f'mu={mus[i]}')
    
    plt.legend()
    
    # save accuracy figure
    if len(acc_data) == 2:
        figure.savefig(f'acc_mu{mus[0]}.png')
    else:
        figure.savefig('acc_all.png')
    
    plt.clf()
    
    # plot mu=0 first
    figure = plt.figure()
    plt.title("FedProx Test Loss vs. Rounds (E=4)")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.plot(X, loss_data[-1], label=f'mu={mus[-1]}')
    
    for i in range(len(mus)-1):
        plt.plot(X, loss_data[i], label=f'mu={mus[i]}')
    
    plt.legend()
    
    # save loss figure
    if len(loss_data) == 2:
        figure.savefig(f'loss_mu{mus[0]}.png')
    else:
        figure.savefig('loss_all.png')
    
    plt.clf()

# save all
save_figure(acc_data, loss_data, mus)

# save pairs
for i in range(len(mus)-1):
    temp_acc_data = [acc_data[i], acc_data[-1]]
    temp_loss_data = [loss_data[i], loss_data[-1]]
    temp_mus = [mus[i], mus[-1]]
    save_figure(temp_acc_data, temp_loss_data, temp_mus)
