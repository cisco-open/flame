/*
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
*/

#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>

#define MAX_SOCK_MAP_MAP_ENTRIES 65535

struct bpf_map_def SEC("maps") sock_map = {
        .type = BPF_MAP_TYPE_SOCKMAP,
        .key_size = sizeof(int),
        .value_size = sizeof(int),
        .max_entries = MAX_SOCK_MAP_MAP_ENTRIES,
        .map_flags = 0
};

SEC("sk_msg_tx")
int bpf_skmsg_tx(struct sk_msg_md *msg)
{
    char* data = msg->data;
    char* data_end = msg->data_end;
    bpf_printk("[sk_msg_tx] get a skmsg of length %d", msg->size);

    if(data + 4 > data_end) {
        return SK_DROP;
    }
    int next_agg_id = *((int*)data);
    bpf_printk("[sk_msg] redirect to socket of aggregator %d", next_agg_id);

    int ret = 0;
    ret = bpf_msg_redirect_map(msg, &sock_map, next_agg_id, BPF_F_INGRESS);

    bpf_printk("TRY REDIRECT TO AGG#%d", next_agg_id);
    if (ret != SK_PASS)
        bpf_printk("REDIRECT TO AGG#%d FAILED", next_agg_id);

    return ret;
}

char _license[] SEC("license") = "GPL";