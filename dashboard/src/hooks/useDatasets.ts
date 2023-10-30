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