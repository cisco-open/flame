import { useToast } from "@chakra-ui/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LOGGEDIN_USER } from "../../../constants";
import { Job } from "../../../entities/Job";
import ApiClient from "../../../services/api-client";

const apiClient = new ApiClient<Job[]>(`users/${LOGGEDIN_USER.name}/jobs`);

const useJobs = (id?: string, onClose?: () => void) => {
    const jobStatusApiClient = new ApiClient<Job[]>(`users/${LOGGEDIN_USER.name}/jobs/${id}/status`);
    const jobsApiClient = new ApiClient<Job[]>(`users/${LOGGEDIN_USER.name}/jobs/${id}`);

    const toast = useToast();
    const queryClientHook = useQueryClient();

    const queryClient =  useQuery({
        queryKey: ['jobs'],
        queryFn: apiClient.getAll,
    });

    const deleteMutation = useMutation({
       mutationFn: jobsApiClient.deleteWithoutParam,
       onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['jobs'] });
            toast({
                title: 'Job successfully deleted',
                status: 'success',
            });
       }
    });

    const createMutation = useMutation({
        mutationFn: apiClient.post,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['jobs'] });
            toast({
                title: 'Job successfully created',
                status: 'success',
            });

            if (onClose) { onClose() }
        }
    });

    const editMutation = useMutation({
        mutationFn: jobsApiClient.put,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['jobs'] });
            toast({
                title: 'Job successfully updated',
                status: 'success',
            });
            if (onClose) { onClose() }
        }
    });

    const updateStatusMutation = useMutation({
        mutationFn: jobStatusApiClient.put,
        onSuccess: () => {
            queryClientHook.invalidateQueries({ queryKey: ['jobs'] });
            toast({
                title: 'Job status successfully updated',
                status: 'success',
            })
        },
    });

    return { ...queryClient, createMutation, updateStatusMutation, editMutation, deleteMutation }
}

export default useJobs;