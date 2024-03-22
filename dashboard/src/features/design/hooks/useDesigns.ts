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
import { Design } from "../../../entities/Design";
import ApiClient from "../../../services/api-client";
import { AxiosError } from "axios";

const apiClient = new ApiClient<Design[]>(`users/${LOGGEDIN_USER.name}/designs`);

const useDesigns = (data?: any) => {
    const updateApiClient = new ApiClient<any>(`users/${LOGGEDIN_USER.name}/designs/${data?.designInEdit?.id}`);

    const queryClientHook = useQueryClient();
    const toast = useToast();

    const createMutation = useMutation({
        mutationFn: apiClient.post,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['designs'] });
            toast({
                title: 'Design successfully created',
                status: 'success',
            })
        },
    });

    const updateMutation = useMutation({
        mutationFn: updateApiClient.put,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['designs'] });
            queryClientHook.invalidateQueries(['design', data?.designInEdit?.id]);
            data?.onClose();
            toast({
                title: 'Design successfully updated',
                status: 'success',
            })
        },
        onError: (error: AxiosError) => {
            toast({
                title: `${error?.response?.data || 'An error occured.'}` ,
                status: 'error',
            })
        },
    });

    const forceDeleteMutation = useMutation({
        mutationFn: apiClient.deleteWithQueryParams,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['designs'] });

            if (data?.navigate) {
                data.navigate('/design');
            }

            toast({
                title: 'Design successfully deleted',
                status: 'success',
            })
        },
    });

    const deleteMutation = useMutation({
        mutationFn: apiClient.delete,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['designs'] });

            if (data?.navigate) {
                data.navigate('/design');
            }

            toast({
                title: 'Design successfully deleted',
                status: 'success',
            })
        },
        onError: (err: AxiosError) => {
            if ((err?.response?.data as string)?.includes('design used in job')) {
                data.onForceDeleteOpen();
            }
        }
    });

    const queryClient = useQuery({
        queryKey: ['designs'],
        queryFn: apiClient.getAll,
    });

    return { ...queryClient, createMutation, deleteMutation, updateMutation, forceDeleteMutation }
}

export default useDesigns;