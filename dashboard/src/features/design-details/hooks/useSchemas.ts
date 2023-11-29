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
import { Schema } from "yup";
import { LOGGEDIN_USER } from "../../../constants";
import ApiClient from "../../../services/api-client";


const useSchema = (id: string) => {
    const apiClient = new ApiClient<Schema[]>(`users/${LOGGEDIN_USER.name}/designs/${id}/schema`);
    const queryClientHook = useQueryClient();
    const toast = useToast();

    const updateMutation = useMutation({
        mutationFn: apiClient.post,
        onSuccess: () => {
            toast({
                title: 'Design schema successfully updated',
                status: 'success',
            })
        },
    });

    const deleteMutation = useMutation({
        mutationFn: apiClient.deleteWithoutParam,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['design'] });
            toast({
                title: 'Design schema successfully removed',
                status: 'success',
            })
        },
    });

    const query = useQuery({
        enabled: !!id,
        queryKey: ['schema', id],
        queryFn: () => apiClient.getAll,
    })

    return { ...query, updateMutation, deleteMutation };
};

export default useSchema;