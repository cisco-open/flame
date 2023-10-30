import { useQuery } from "@tanstack/react-query";
import { LOGGEDIN_USER } from "../../../constants";
import { Task } from "../../../entities/Task";
import ApiClient from "../../../services/api-client";

const useJob = (id: string) => {
    const apiClient = new ApiClient<Task[]>(`users/${LOGGEDIN_USER.name}/jobs`);
    return useQuery({
        enabled: !!id,
        queryKey: ['job', id],
        queryFn: () => apiClient.get(id),
    });
}

export default useJob;