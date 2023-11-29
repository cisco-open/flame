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

import axios, { AxiosInstance, AxiosRequestConfig } from "axios";
import { BASE_URL, ML_FLOW_BASE_URL } from '../environment';

class ApiClient<T> {
    endpoint: string;
    axiosInstance: AxiosInstance;

    constructor(endpoint: string, isMlFlow?: boolean) {
        this.endpoint = endpoint;
        const API_URL = (window as unknown as any).env;
        this.axiosInstance = axios.create({
            baseURL: `${isMlFlow ?
                process.env.NODE_ENV === 'development' ? ML_FLOW_BASE_URL :  API_URL?.REACT_APP_MLFLOW_URL :
                process.env.NODE_ENV === 'development' ? BASE_URL : API_URL?.REACT_APP_API_URL}`,
        })
    }

    getAll = (params?: AxiosRequestConfig) => this.axiosInstance.get<T>(this.endpoint, params)
        .then(res => res.data);

    get = (id: string) => this.axiosInstance.get<T>(this.endpoint + '/' + id).then(res => res.data);

    post = (payload: any) => this.axiosInstance.post<T>(this.endpoint, payload).then(res => res.data);

    delete = (id?: string) => this.axiosInstance.delete<T>(`${this.endpoint}/${id}`);

    deleteWithoutParam = () => this.axiosInstance.delete<T>(`${this.endpoint}`);

    put = (payload: any) => this.axiosInstance.put<T>(`${this.endpoint}`, payload);

    pushFile = (data: any) => {
        const formData = new FormData();
        formData.append('fileData', data.fileData);
        formData.append('fileName', data.fileName);

        return this.axiosInstance.post(this.endpoint, { fileName: data.fileName, fileData: data.fileData }, { headers: { 'Content-Type': 'multipart/form-data' } })
    }
}

export default ApiClient;