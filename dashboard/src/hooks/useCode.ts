import { useToast } from "@chakra-ui/react";
import { useMutation, useQuery } from "@tanstack/react-query";
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