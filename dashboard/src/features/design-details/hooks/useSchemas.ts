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