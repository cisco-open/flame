/**
 * Copyright 2024 Cisco Systems, Inc. and its affiliates
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
import ApiClient from "../../../services/api-client";

const useComputes = (onClose?: () => void) => {
    const apiClient = new ApiClient<any[]>('computes');

    const toast = useToast();
    const queryClientHook = useQueryClient();

    const queryClient =  useQuery({
        queryKey: ['computes'],
        queryFn: apiClient.getAll,
    });

    const deleteMutation = useMutation({
       mutationFn: apiClient.delete,
       onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['computes'] });
            toast({
                title: 'Compute successfully deleted',
                status: 'success',
            });
       }
    });

    const createMutation = useMutation({
        mutationFn: apiClient.post,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['computes'] });
            toast({
                title: 'Compute successfully created',
                status: 'success',
            });

            if (onClose) { onClose() }
        }
    });

    const editMutation = useMutation({
        mutationFn: apiClient.put,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['computes'] });
            toast({
                title: 'Compute successfully updated',
                status: 'success',
            });
            if (onClose) { onClose() }
        }
    });

    return { ...queryClient, createMutation, editMutation, deleteMutation }
}

export default useComputes;