import axios, { AxiosInstance, AxiosRequestConfig } from "axios";
import { BASE_URL, ML_FLOW_BASE_URL } from '../environment';

class ApiClient<T> {
    endpoint: string;
    axiosInstance: AxiosInstance;

    constructor(endpoint: string, isMlFlow?: boolean) {
        this.endpoint = endpoint;

        this.axiosInstance = axios.create({
            baseURL: `${isMlFlow ? ML_FLOW_BASE_URL : BASE_URL}`,
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