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
import { AxiosError } from "axios";
import { LOGGEDIN_USER } from "../constants";
import { Dataset } from "../entities/Dataset";
import ApiClient from "../services/api-client";

const apiClient = new ApiClient<Dataset[]>('datasets');
const mutateApiClient = new ApiClient<Dataset>(`users/${LOGGEDIN_USER.name}/datasets`);

const useDatasets = (data?: any) => {
    const queryClientHook = useQueryClient();
    const toast = useToast();
    const createMutation = useMutation({
        mutationFn: mutateApiClient.post,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['datasets'] });
            toast({
                title: 'Dataset successfully created',
                status: 'success',
            });
            data?.onClose();
            data?.setIsSaveSuccess(true);
        },
        onError: (error: AxiosError) => {
            data?.setIsSaveSuccess(false);
            toast({
                title: `${error?.response?.data || 'An error occured.'}` ,
                status: 'error',
            });
        }
    });
    const query = useQuery({
        queryKey: ['datasets'],
        queryFn: apiClient.getAll,
    });

    return { ...query, createMutation }
}

export default useDatasets;