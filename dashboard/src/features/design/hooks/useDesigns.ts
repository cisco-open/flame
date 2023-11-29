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

const apiClient = new ApiClient<Design[]>(`users/${LOGGEDIN_USER.name}/designs`);

const useDesigns = () => {
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

    const deleteMutation = useMutation({
        mutationFn: apiClient.delete,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['designs'] });
            toast({
                title: 'Design successfully deleted',
                status: 'success',
            })
        },
    });

    const queryClient = useQuery({
        queryKey: ['designs'],
        queryFn: apiClient.getAll,
    });

    return { ...queryClient, createMutation, deleteMutation }
}

export default useDesigns;