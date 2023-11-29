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
import { useMutation } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { LOGGEDIN_USER } from "../constants";
import ApiClient from "../services/api-client";


const useCode = (designId: string) => {
    const apiClient = new ApiClient<any>(`users/${LOGGEDIN_USER.name}/designs/${designId}/code`);
    const toast = useToast();

    const deleteCodeMutation = useMutation({
        mutationFn: apiClient.deleteWithoutParam,
    });

    const pushFileMutation = useMutation({
        mutationFn: apiClient.pushFile,
        onSuccess: () => {
            toast({
                title: 'Design code file successfully updated',
                status: 'success',
            })
        },
        onError: (error: AxiosError) => {
            toast({
                title: error.message,
                status: 'error',
            });
        }
    });

    return { pushFileMutation, deleteCodeMutation }
}

export default useCode;