/**
 * Copyright 2023 Cisco Systems, Inc. and its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { useToast } from "@chakra-ui/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LOGGEDIN_USER } from "../../../constants";
import { Job } from "../../../entities/Job";
import ApiClient from "../../../services/api-client";
import { NavigateFunction } from "react-router-dom";
import { AxiosError } from "axios";

const apiClient = new ApiClient<Job[]>(`users/${LOGGEDIN_USER.name}/jobs`);

const useJobs = (id?: string, onClose?: () => void, navigate?: NavigateFunction | undefined) => {
    const jobStatusApiClient = new ApiClient<Job[]>(`users/${LOGGEDIN_USER.name}/jobs/${id}/status`);
    const jobsApiClient = new ApiClient<Job[]>(`users/${LOGGEDIN_USER.name}/jobs/${id}`);

    const toast = useToast();
    const queryClientHook = useQueryClient();

    const queryClient =  useQuery({
        queryKey: ['jobs'],
        queryFn: apiClient.getAll,
    });

    const deleteMutation = useMutation({
       mutationFn: jobsApiClient.deleteWithoutParam,
       onSuccess: () => {
            queryClientHook.invalidateQueries(['jobs']);
            queryClientHook.invalidateQueries(['jobStatus', id]);

            toast({
                title: 'Job successfully deleted',
                status: 'success',
            });

            if (navigate) {
                navigate('/jobs');
            }
       }
    });

    const createMutation = useMutation({
        mutationFn: apiClient.post,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['jobs'] });
            toast({
                title: 'Job successfully created',
                status: 'success',
            });

            if (onClose) { onClose() }
        }
    });

    const editMutation = useMutation({
        mutationFn: jobsApiClient.put,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['jobs'] });
            queryClientHook.invalidateQueries(['job', id]);
            toast({
                title: 'Job successfully updated',
                status: 'success',
            });
            if (onClose) { onClose() }
        }
    });

    const updateStatusMutation = useMutation({
        mutationFn: jobStatusApiClient.put,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['jobs'] });
            queryClientHook.invalidateQueries(['jobStatus', id]);
            toast({
                title: 'Job status successfully updated',
                status: 'success',
            })
        },
        onError: (err: AxiosError) => {
            toast({
                title: `${err?.response?.data}`,
                status: 'error',
            })
        },
    });

    return { ...queryClient, createMutation, updateStatusMutation, editMutation, deleteMutation }
}

export default useJobs;