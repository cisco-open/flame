# Copyright 2024 Cisco Systems, Inc. and its affiliates
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

"""
This script is used to partition the FedScale dataset into multiple clients.
"""

import logging
import os
import sys
import torchvision.models as tormodels
from torchvision import datasets, transforms
import argparse
import csv

from fedscale.dataloaders.utils_data import get_data_transform
from fedscale.utils.models.torch_model_provider import get_cv_model
from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset

tokenizer = None

def import_libs():
    global tokenizer

    if args.task == 'nlp' or args.task == 'text_clf':
        global AdamW, AlbertTokenizer, AutoConfig, AutoModelWithLMHead, AutoTokenizer, MobileBertForPreTraining, load_and_cache_examples, mask_tokens

        from transformers import (AdamW, AlbertTokenizer, AutoConfig,
                                  AutoModelWithLMHead, AutoTokenizer,
                                  MobileBertForPreTraining)

        from fedscale.dataloaders.nlp import load_and_cache_examples, mask_tokens
        tokenizer = AlbertTokenizer.from_pretrained(
            'albert-base-v2', do_lower_case=True)
    elif args.task == 'speech':
        global numba, SPEECH, BackgroundNoiseDataset, AddBackgroundNoiseOnSTFT, DeleteSTFT, FixSTFTDimension, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, ToMelSpectrogramFromSTFT, ToSTFT, ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, LoadAudio, ToMelSpectrogram, ToTensor

        import numba

        from fedscale.dataloaders.speech import SPEECH, BackgroundNoiseDataset
        from fedscale.dataloaders.transforms_stft import (AddBackgroundNoiseOnSTFT,
                                                          DeleteSTFT,
                                                          FixSTFTDimension,
                                                          StretchAudioOnSTFT,
                                                          TimeshiftAudioOnSTFT,
                                                          ToMelSpectrogramFromSTFT,
                                                          ToSTFT)
        from fedscale.dataloaders.transforms_wav import (ChangeAmplitude,
                                                         ChangeSpeedAndPitchAudio,
                                                         FixAudioLength, LoadAudio,
                                                         ToMelSpectrogram,
                                                         ToTensor)

def init_dataset(args):
    import_libs()

    if args.data_set == 'Mnist':
        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                        transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                        transform=test_transform)

    elif args.data_set == 'cifar10':
        train_transform, test_transform = get_data_transform('cifar')
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                            transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                        transform=test_transform)

    elif args.data_set == "imagenet":
        train_transform, test_transform = get_data_transform('imagenet')
        train_dataset = datasets.ImageNet(
            args.data_dir, split='train', download=False, transform=train_transform)
        test_dataset = datasets.ImageNet(
            args.data_dir, split='val', download=False, transform=test_transform)

    elif args.data_set == 'emnist':
        test_dataset = datasets.EMNIST(
            args.data_dir, split='balanced', train=False, download=True, transform=transforms.ToTensor())
        train_dataset = datasets.EMNIST(
            args.data_dir, split='balanced', train=True, download=True, transform=transforms.ToTensor())

    elif args.data_set == 'femnist':
        from fedscale.dataloaders.femnist import FEMNIST

        train_transform, test_transform = get_data_transform('mnist')
        train_dataset = FEMNIST(
            args.data_dir, dataset='train', transform=train_transform)
        test_dataset = FEMNIST(
            args.data_dir, dataset='test', transform=test_transform)

    elif args.data_set == 'openImg':
        from fedscale.dataloaders.openimage import OpenImage

        train_transform, test_transform = get_data_transform('openImg')
        train_dataset = OpenImage(
            args.data_dir, dataset='train', transform=train_transform)
        test_dataset = OpenImage(
            args.data_dir, dataset='test', transform=test_transform)

    elif args.data_set == 'reddit':
        train_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=False)
        test_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=True)

    elif args.data_set == 'stackoverflow':
        from fedscale.dataloaders.stackoverflow import stackoverflow

        train_dataset = stackoverflow(args.data_dir, train=True)
        test_dataset = stackoverflow(args.data_dir, train=False)

    elif args.data_set == 'amazon':
        if args.model == 'albert':
            import fedscale.dataloaders.amazon as fl_loader
            train_dataset = fl_loader.AmazonReview_loader(
                args.data_dir, train=True, tokenizer=tokenizer, max_len=args.clf_block_size)
            test_dataset = fl_loader.AmazonReview_loader(
                args.data_dir, train=False, tokenizer=tokenizer, max_len=args.clf_block_size)

        elif args.model == 'lr':
            import fedscale.dataloaders.word2vec as fl_loader
            train_dataset = fl_loader.AmazonReview_word2vec(
                args.data_dir, args.embedding_file, train=True)
            test_dataset = fl_loader.AmazonReview_word2vec(
                args.data_dir, args.embedding_file, train=False)

    elif args.data_set == 'yelp':
        import fedscale.dataloaders.yelp as fl_loader

        train_dataset = fl_loader.TextSentimentDataset(
            args.data_dir, train=True, tokenizer=tokenizer, max_len=args.clf_block_size)
        test_dataset = fl_loader.TextSentimentDataset(
            args.data_dir, train=False, tokenizer=tokenizer, max_len=args.clf_block_size)

    elif args.data_set == 'google_speech':
        print("Loading Google Speech dataset...")
        bkg = '_background_noise_'
        data_aug_transform = transforms.Compose(
            [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
                TimeshiftAudioOnSTFT(), FixSTFTDimension()])
        bg_dataset = BackgroundNoiseDataset(
            os.path.join(args.data_dir, bkg), data_aug_transform)
        add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
        train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(
            n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
        train_dataset = SPEECH(args.data_dir, dataset='train',
                                transform=transforms.Compose([LoadAudio(),
                                                                data_aug_transform,
                                                                add_bg_noise,
                                                                train_feature_transform]))
        valid_feature_transform = transforms.Compose(
            [ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
        test_dataset = SPEECH(args.data_dir, dataset='test',
                                transform=transforms.Compose([LoadAudio(),
                                                            FixAudioLength(),
                                                            valid_feature_transform]))
        print("Google Speech dataset loaded!")
    elif args.data_set == 'common_voice':
        from fedscale.dataloaders.voice_data_loader import \
            SpectrogramDataset
        train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                            data_dir=args.data_dir,
                                            labels=model.labels,
                                            train=True,
                                            normalize=True,
                                            speed_volume_perturb=args.speed_volume_perturb,
                                            spec_augment=args.spec_augment,
                                            data_mapfile=args.data_mapfile)
        test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                            data_dir=args.data_dir,
                                            labels=model.labels,
                                            train=False,
                                            normalize=True,
                                            speed_volume_perturb=False,
                                            spec_augment=False)
    else:
        logging.info('DataSet must be {}!'.format(
            ['Mnist', 'Cifar', 'openImg', 'reddit', 'stackoverflow', 'speech', 'yelp']))
        sys.exit(-1)

    return train_dataset, test_dataset

"""
Example 1 (Google Speech):
python dataset_partitioner.py --data_set google_speech --data_dir ./FedScale/benchmark/dataset/data/google_speech/ --task speech --model resnet34 --model_zoo none --num_participants 500

Example 2 (FEMNIST):
python dataset_partitioner.py --data_set femnist --data_dir ./FedScale/benchmark/dataset/data/femnist/ --task femnist --model resnet152 --model_zoo none --num_participants 3400

Example 3 (Reddit):
python dataset_partitioner.py --data_set reddit --data_dir ./FedScale/benchmark/dataset/data/reddit/ --task nlp --model albert-base-v2 --model_zoo none --num_participants 130000
"""

parser = argparse.ArgumentParser()

parser.add_argument('--data_set', type=str, help='dataset name')
parser.add_argument('--data_dir', type=str, help='dataset directory')
parser.add_argument('--task', type=str, help='task name')
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--model_zoo', type=str, help='model zoo')
parser.add_argument('--num_participants', type=int, help='# of participants')
parser.add_argument('--data_map_file', type=int, help='path to client-to-data mapping file')
parser.add_argument('--block_size', default=64, help='block_size')
parser.add_argument('--overwrite_cache', default=False, help='block_size')

args = parser.parse_args()

print('dataset name:', args.data_set)
print('dataset directory:', args.data_dir)

# Load dataset from file system
train_dataset, test_dataset = init_dataset(args)

print(f"instances in train dataset: {len(train_dataset)}")
print(f"instances in test dataset: {len(test_dataset)}")

outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47, 'amazon': 5,
               'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5, 'inaturalist': 1010
               }

num_class = 0
train_test_ratio = 0
if args.data_set == "google_speech":
    num_class = outputClass['google_speech']
    train_test_ratio = 50
elif args.data_set == "femnist":
    num_class = outputClass['femnist']
    train_test_ratio = 50
elif args.data_set == "reddit":
    num_class = 0
    train_test_ratio = 53
    args.overwrite_cache = False
    args.block_size = 64
    args.model = None

print(f"Number of outputClass in {args.data_set} dataset: {num_class}")

print("Data partitioner starts ...")

training_sets = DataPartitioner(data=train_dataset, args=args, numOfClass=num_class)
print("Run partition_data_helper for training_sets ...")
training_sets.partition_data_helper(num_clients=args.num_participants, data_map_file=args.data_map_file)

testing_sets = DataPartitioner(data=test_dataset, args=args, numOfClass=num_class, isTest=True)
testing_sets.partition_data_helper(num_clients=int(args.num_participants/train_test_ratio) + 1)

print(f"Number of partitions: {len(training_sets.partitions)}.")

# print(f"partition[0]: {training_sets.partitions[0]}.")
# print(f"partition[0][0]: {training_sets.partitions[0][0]}.")

# print(f"Train dataset raw data: {train_dataset.data[training_sets.partitions[0][0]]}.")
# print(f"Train dataset raw tag: {train_dataset.targets[training_sets.partitions[0][0]]}.")

print("Data partitioner completes ...")

def read_lines_and_write_to_csv(csv_reader_dict, output_filename, lines):
    with open(output_filename, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        if args.data_set == "reddit":
            csv_writer.writerow(lines)
        else:
            for line_index in lines:
                if line_index >= 0 and line_index in csv_reader_dict:
                    csv_writer.writerow(csv_reader_dict[line_index])

print("\n+++++ Writing training partitions to CSV files ... +++++")
if not os.path.exists("/tmp/flame_dataset/" + args.data_set + "/train"):
    os.makedirs("/tmp/flame_dataset/" + args.data_set + "/train")

train_csv_path = os.path.join(args.data_dir, 'client_data_mapping', 'train.csv')
completed = 0
with open(train_csv_path, 'r') as input_file:
    print("\n Reading training CSV files ...")
    csv_reader = csv.reader(input_file)
    csv_reader_dict = {}
    for index, row in enumerate(csv_reader):
        csv_reader_dict[index] = row
    print("\n Reading training CSV files is done ...")

    for i in range(len(training_sets.partitions)):
        output_filename = "client-" + str(i) + "-train.csv"
        output_path = os.path.join("/tmp/flame_dataset/", args.data_set, "train", output_filename)
        read_lines_and_write_to_csv(csv_reader_dict, output_path, training_sets.partitions[i])

        if i % 100 == 0:
            completed += 100
            print(f"{completed} partitions completed, {len(training_sets.partitions) - completed} remains...")

print("\n+++++ Writing testing partitions to CSV files ... +++++")
if not os.path.exists("/tmp/flame_dataset/" + args.data_set + "/test"):
    os.makedirs("/tmp/flame_dataset/" + args.data_set + "/test")

test_csv_path = os.path.join(args.data_dir, 'client_data_mapping', 'test.csv')
completed = 0
with open(test_csv_path, 'r') as input_file:
    print("\n Reading testing CSV files ...")
    csv_reader = csv.reader(input_file)
    csv_reader_dict = {}
    for index, row in enumerate(csv_reader):
        csv_reader_dict[index] = row
    print("\n Reading testing CSV files is done ...")

    for i in range(len(testing_sets.partitions)):
        output_filename = "client-" + str(i) + "-test.csv"
        output_path = os.path.join("/tmp/flame_dataset/", args.data_set, "test", output_filename)
        read_lines_and_write_to_csv(csv_reader_dict, output_path, testing_sets.partitions[i])

        if i % 10 == 0:
            completed += 10
            print(f"{completed} partitions completed, {len(testing_sets.partitions) - completed} remains...")