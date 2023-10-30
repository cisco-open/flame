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