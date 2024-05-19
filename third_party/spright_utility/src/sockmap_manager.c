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

#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef likely
#define likely(x)   __builtin_expect(!!(x), 1)
#endif /* likely */

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif /* unlikely */

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif /* SYS_pidfd_open */

#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif /* SYS_pidfd_getfd */

#define MAP_NAME "sock_map"

int PORT_SK_MSG = 10105;
int PORT_RPC = 10106;

/* TODO: Cleanup on errors */
static void *dummy_server(void* arg)
{
    struct sockaddr_in addr;
    int sockfd_l;
    int sockfd_c;
    int optval;
    int ret;

    sockfd_l = socket(AF_INET, SOCK_STREAM, 0);
    if (unlikely(sockfd_l == -1)) {
        fprintf(stderr, "socket() error: %s\n", strerror(errno));
        pthread_exit(NULL);
    }

    optval = 1;
    ret = setsockopt(sockfd_l, SOL_SOCKET, SO_REUSEADDR, &optval,
                     sizeof(int));
    if (unlikely(ret == -1)) {
        fprintf(stderr, "setsockopt() error: %s\n", strerror(errno));
        pthread_exit(NULL);
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT_SK_MSG);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    ret = bind(sockfd_l, (struct sockaddr *)&addr,
               sizeof(struct sockaddr_in));
    if (unlikely(ret == -1)) {
        fprintf(stderr, "bind() error: %s\n", strerror(errno));
        pthread_exit(NULL);
    }

    /* TODO: Correct backlog? */
    ret = listen(sockfd_l, 10);
    if (unlikely(ret == -1)) {
        fprintf(stderr, "listen() error: %s\n", strerror(errno));
        pthread_exit(NULL);
    }

    while (1) {
        sockfd_c = accept(sockfd_l, NULL, NULL);
        if (unlikely(sockfd_c == -1)) {
            fprintf(stderr, "accept() error: %s\n",
                    strerror(errno));
            pthread_exit(NULL);
        }
    }

    pthread_exit(NULL);
}

static int rpc_server(int fd_sk_msg_map)
{
    struct sockaddr_in addr;
    ssize_t bytes_received;
    int sockfd_sk_msg_nf;
    int buffer[3];
    int sockfd_l;
    int sockfd_c;
    int optval;
    int pidfd;
    int ret;

    sockfd_l = socket(AF_INET, SOCK_STREAM, 0);
    if (unlikely(sockfd_l == -1)) {
        fprintf(stderr, "socket() error: %s\n", strerror(errno));
        return -1;
    }

    optval = 1;
    ret = setsockopt(sockfd_l, SOL_SOCKET, SO_REUSEADDR, &optval,
                     sizeof(int));
    if (unlikely(ret == -1)) {
        fprintf(stderr, "setsockopt() error: %s\n", strerror(errno));
        return -1;
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT_RPC);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    ret = bind(sockfd_l, (struct sockaddr *)&addr,
               sizeof(struct sockaddr_in));
    if (unlikely(ret == -1)) {
        fprintf(stderr, "bind() error: %s\n", strerror(errno));
        return -1;
    }

    /* TODO: Correct backlog? */
    ret = listen(sockfd_l, 10);
    if (unlikely(ret == -1)) {
        fprintf(stderr, "listen() error: %s\n", strerror(errno));
        return -1;
    }

    for (;;) {
        sockfd_c = accept(sockfd_l, NULL, NULL);
        if (unlikely(sockfd_c == -1)) {
            fprintf(stderr, "accept() error: %s\n",
                    strerror(errno));
            return -1;
        }

        bytes_received = recv(sockfd_c, buffer, 3 * sizeof(int), 0);
        if (unlikely(bytes_received == -1)) {
            fprintf(stderr, "recv() error: %s\n", strerror(errno));
            return -1;
        }

        printf("SK_MSG metadata: PID %d; socket FD %d; Fn ID %d\n", buffer[0], buffer[1], buffer[2]);

        pidfd = syscall(SYS_pidfd_open, buffer[0], 0);
        if (unlikely(ret == -1)) {
            fprintf(stderr, "SYS_pidfd_open() error: %s\n",
                    strerror(errno));
            return -1;
        }

        sockfd_sk_msg_nf = syscall(SYS_pidfd_getfd, pidfd, buffer[1],
                                   0);
        if (unlikely(ret == -1)) {
            fprintf(stderr, "__NR_pidfd_getfd() error: %s\n",
                    strerror(errno));
            return -1;
        }

        ret = bpf_map_update_elem(fd_sk_msg_map, &buffer[2],
                                  &sockfd_sk_msg_nf, 0);
        if (unlikely(ret < 0)) {
            fprintf(stderr, "bpf_map_update_elem() error: %s\n",
                    strerror(-ret));
            return -1;
        }

        printf("%s: NF_ID %d -> SOCKFD %d\n", MAP_NAME, buffer[2],
               sockfd_sk_msg_nf);

        ret = close(sockfd_c);
        if (unlikely(ret == -1)) {
            fprintf(stderr, "close() error: %s\n", strerror(errno));
            return -1;
        }
    }

    ret = close(sockfd_l);
    if (unlikely(ret == -1)) {
        fprintf(stderr, "close() error: %s\n", strerror(errno));
        return -1;
    }

    return 0;
}

static int init_sockmap_mgr(void)
{
    printf("Initializing sockmap manager... Listening on PORT_SK_MSG[%d] and PORT_RPC[%d]\n", PORT_SK_MSG, PORT_RPC);

    struct bpf_object* obj = NULL;
    int fd_sk_msg_prog;
    int fd_sk_msg_map;
    pthread_t thread;
    int ret;

    printf("Initializing SK_MSG server...\n");
    ret = pthread_create(&thread, NULL, &dummy_server, NULL);
    if (unlikely(ret != 0)) {
        fprintf(stderr, "pthread_create() error: %s\n", strerror(ret));
        return -1;
    }

    printf("Loading eBPF programs...\n");
    ret = bpf_prog_load("ebpf/sk_msg_kern.o", BPF_PROG_TYPE_SK_MSG, &obj,
                        &fd_sk_msg_prog);
    if (unlikely(ret < 0)) {
        fprintf(stderr, "bpf_prog_load() error: %s\n", strerror(-ret));
        return -1;
    }

    printf("Loading sockmap...\n");
    fd_sk_msg_map = bpf_object__find_map_fd_by_name(obj, MAP_NAME);
    if (unlikely(fd_sk_msg_map < 0)) {
        fprintf(stderr, "bpf_object__find_map_fd_by_name() error: %s\n",
                strerror(-ret));
        return -1;
    }

    printf("Attaching SK_MSG program to sockmap...\n");
    ret = bpf_prog_attach(fd_sk_msg_prog, fd_sk_msg_map, BPF_SK_MSG_VERDICT,
                          0);
    if (unlikely(ret < 0)) {
        fprintf(stderr, "bpf_prog_attach() error: %s\n",
                strerror(-ret));
        return -1;
    }

    printf("Starting RPC server...\n");
    ret = rpc_server(fd_sk_msg_map);
    if (unlikely(ret == -1)) {
        fprintf(stderr, "rpc_server() error\n");
        return -1;
    }

    return 0;
}

static int exit_sockmap_mgr(void)
{
    return 0;
}

static void stopper(void)
{
    while (1) {
        sleep(30);
    }
}

static int sockmap_mgr(void)
{
    int ret;

    ret = init_sockmap_mgr();
    if (unlikely(ret == -1)) {
        fprintf(stderr, "init_sockmap_mgr() error\n");
        return -1;
    }

    stopper();

    ret = exit_sockmap_mgr();
    if (unlikely(ret == -1)) {
        fprintf(stderr, "exit_sockmap_mgr() error\n");
        return -1;
    }

    return 0;
}

void help() {
    printf("Usage: sudo ./sockmap_manager [options]\n");
    printf("Options:\n");
    printf("  -h, --help           Show this help message\n");
    printf("  -s, --sk-msg-port    Set the SK_MSG port\n");
    printf("  -r, --rpc-port       Set the RPC port\n");
}

int main(int argc, char **argv)
{

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            help();
            return 0;
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sk-msg-port") == 0) && i + 1 < argc) {
            PORT_SK_MSG = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rpc-port") == 0) && i + 1 < argc) {
            PORT_RPC = atoi(argv[++i]);
        } else {
            printf("Unknown option: %s\n", argv[i]);
            help();
            return 1;
        }
    }

    int ret;

    ret = sockmap_mgr();
    if (unlikely(ret == -1)) {
        fprintf(stderr, "sockmap_mgr() error\n");
        return -1;
    }

    return 0;
}